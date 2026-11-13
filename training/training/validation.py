"""Validation Engine for LemGendary Training Suite.

Executes deterministic evaluation passes under torch.no_grad(), computing task-specific
loss metrics, PSNR/SSIM reconstruction fidelity, or classification scores. The metric
families live in :mod:`training.training.val_metrics` (per batch) and
:mod:`training.training.val_summary` (end of pass); this module drives the loop: resume
fast-forward, forward passes, OOM recovery, progress checkpoints and telemetry.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import torch

from training.training.amp import get_autocast_context
from training.training.context import TrainingContext
from training.training.epoch import _unpack_batch
from training.training.step_progress import (
    BatchClock,
    batch_progress_line,
    is_main_process,
    is_oom_error,
    progress_payload,
    release_cuda_cache,
    resume_iteration,
)
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import BATCH_METRICS, LIGHT_BATCH_METRICS, ValBatch
from training.training.val_summary import is_llm, summarize

logger = logging.getLogger("lemtrain.training.validation")

# Per-epoch validation runs LPIPS and FID on every Nth batch; the final test pass, SWA
# evaluation and standalone evaluation keep every batch (stride 1).
EPOCH_HEAVY_METRIC_STRIDE = 4


def _log_due(clock: BatchClock, step: int, now: float) -> bool:
    """True on warm-up steps, about every 1% of the pass, the last step, or after an 8 s gap."""
    log_interval = max(1, clock.num_batches // 100) if clock.num_batches > 100 else 10
    is_warmup = step in (1, 2, 5, 10)
    is_periodic = (step % log_interval == 0) or (step == clock.num_batches)
    is_heartbeat = now - clock.last_log_time >= 8.0 and step > 10
    return is_warmup or is_periodic or is_heartbeat


class _ValidationPass:
    """One validation pass over ``ctx.data.val_loader`` or an explicit test loader."""

    def __init__(self, ctx: TrainingContext, epoch: int, loader: Any = None) -> None:
        self.ctx = ctx
        self.epoch = epoch
        self.loader = loader if loader is not None else ctx.data.val_loader
        self.acc = ValAccumulators()
        self.step_count = 0
        start_time = time.time()
        num_batches = (
            len(self.loader)
            if self.loader is not None and hasattr(self.loader, "__len__")
            else 0
        )
        self.clock = BatchClock(num_batches, start_time, time.time())

    def resume(self) -> None:
        """Restore accumulators of an interrupted validation pass, if any."""
        resume_step = resume_iteration(
            self.ctx.state.resume_state, "val", self.ctx.state.guards.val_step_handled
        )
        if resume_step <= 0:
            return
        self.clock.resume_step = resume_step
        self.ctx.state.guards.val_step_handled = True
        # Inception features of the skipped batches are not in the checkpoint (review D-13).
        self.acc.image.fid_incomplete = True
        print(
            f"[RESUME] Fast-forwarding {resume_step} completed validation batches in Epoch "
            f"{self.epoch}...",
            flush=True,
        )
        saved = getattr(self.ctx.state.resume_state, "val_accumulators", {})
        if saved:
            self.acc.restore(saved)
            self.step_count = resume_step

    def forward(self, inputs: Any, targets: Any, task_idx: Any) -> tuple[Any, Any]:
        """Return ``(preds, loss)`` for one batch under autocast."""
        ctx = self.ctx
        autocast_ctx = get_autocast_context(
            ctx.hardware.policy, ctx.hardware.device_info.device
        )
        with autocast_ctx:
            eval_step_fn = getattr(ctx.models.model, "eval_step", None)
            if callable(eval_step_fn):
                val_out = eval_step_fn(inputs)
                if isinstance(val_out, dict):
                    return val_out.get("preds", inputs), val_out["loss"]
                return inputs, val_out
            preds = ctx.models.model(inputs)
            if task_idx is not None:
                return preds, ctx.models.criterion(preds, targets, task_idx)
            return preds, ctx.models.criterion(preds, targets)

    def guarded_forward(
        self, batch_idx: int, batch: tuple[Any, Any, Any]
    ) -> tuple[Any, Any] | None:
        """Run :meth:`forward`; on CUDA out-of-memory recover and return None."""
        try:
            return self.forward(*batch)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if not is_oom_error(err):
                raise
            self.relieve_oom(batch_idx)
            return None

    def relieve_oom(self, batch_idx: int) -> None:
        """Let the governor (or a cache flush) recover from an out-of-memory batch."""
        if hasattr(self.ctx.governance.governor, "handle_oom"):
            self.ctx.governance.governor.handle_oom()
        else:
            release_cuda_cache()
        print(
            f"\n  [VAL WARNING] VRAM OOM intercepted on val batch {batch_idx + 1}. "
            "Cache evacuated and recovered.",
            flush=True,
        )

    def save_progress(self) -> None:
        """Offer an intra-epoch progress checkpoint to the lifecycle manager."""
        ctx = self.ctx
        if ctx.governance.lifecycle_manager is None:
            return
        total_val = (
            len(ctx.data.val_loader)
            if ctx.data.val_loader is not None and hasattr(ctx.data.val_loader, "__len__")
            else 0
        )
        ctx.governance.lifecycle_manager.check_and_save_progress(
            epoch=self.epoch,
            phase="val",
            step=self.step_count,
            total_steps=total_val,
            payload_builder=partial(
                _val_payload,
                ctx,
                self.epoch,
                ctx.models.raw_model or ctx.models.model,
                self.acc.snapshot(),
            ),
        )

    def log(self, loss: Any) -> None:
        """Print the batch telemetry line when due (rank 0 only)."""
        clock = self.clock
        step = self.step_count
        now = time.time()
        if not (is_main_process() and _log_due(clock, step, now)):
            return
        clock.last_log_time = now
        rate, elapsed_batch = clock.rate(step, now)
        metrics = (
            f"ValLoss: {float(loss.item()):.4f} | AvgValLoss: {self.acc.total_loss / step:.4f}"
        )
        line = batch_progress_line(
            "VAL BATCH", (step, clock.num_batches), metrics, rate, elapsed_batch
        )
        print(line, end="\n" if step == clock.num_batches else "", flush=True)

    def step(self, batch_idx: int, batch: Any, heavy: bool) -> None:
        """Evaluate one batch; ``heavy`` adds the LPIPS and FID networks to its metrics."""
        unpacked = _unpack_batch(batch, self.ctx.hardware.device_info.device)
        outcome = self.guarded_forward(batch_idx, unpacked)
        if outcome is None:
            return
        preds, loss = outcome
        self.acc.total_loss += float(loss.item())
        self.step_count += 1
        self.save_progress()
        self.log(loss)

        view = self.batch_view(preds, unpacked[1], unpacked[0])
        for update in BATCH_METRICS if heavy else LIGHT_BATCH_METRICS:
            update(self.acc, view)

    def batch_view(self, preds: Any, targets: Any, inputs: Any = None) -> ValBatch:
        """Bundle one batch with the evaluation settings of the model."""
        ctx = self.ctx
        labels = None
        if is_llm(ctx):
            labels = inputs.get("labels") if isinstance(inputs, dict) else targets
        return ValBatch(
            preds=preds,
            targets=targets,
            sota_cfg=ctx.run.model_info.get("sota_targets", {}),
            dataset_type=ctx.run.model_info.get("dataset_type"),
            device=ctx.hardware.device_info.device,
            governor=ctx.governance.governor,
            labels=labels,
        )


def _val_payload(
    ctx: TrainingContext, epoch: int, model: torch.nn.Module, snapshot: dict[str, Any]
) -> dict[str, Any]:
    return progress_payload(
        ctx,
        epoch,
        model,
        {
            "train_metrics": ctx.state.last_train_metrics,
            "val_accumulators": snapshot,
        },
    )


def validate_one_epoch(
    ctx: TrainingContext,
    epoch: int,
    cancel_check: Any = None,
    loader: Any = None,
    heavy_stride: int = 1,
) -> dict[str, float]:
    """Execute validation over the evaluation loader or an explicit loader (e.g. test_loader).

    Args:
        ctx: Training context holding model, validation data loader, and policies.
        epoch: Epoch index being evaluated.
        cancel_check: Optional callable returning True when cancellation is requested.
        loader: Optional explicit DataLoader; defaults to ctx.data.val_loader.
        heavy_stride: LPIPS and FID run on every Nth batch (1 = every batch, exact values;
            larger values give subsampled estimates at a fraction of the cost).

    Returns:
        dict[str, float]: Dictionary of validation metrics.
    """
    heavy_stride = max(1, heavy_stride)
    target_loader = loader if loader is not None else ctx.data.val_loader
    if target_loader is None or len(target_loader) == 0:
        return {"val_loss": 0.0}

    ctx.models.model.eval()
    run = _ValidationPass(ctx, epoch, loader=target_loader)
    run.resume()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, batch in enumerate(target_loader):
            if run.clock.skip_resumed(batch_idx, "VAL BATCH", "validation"):
                continue
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Validation cancelled by user request.")

            run.step(batch_idx, batch, heavy=batch_idx % heavy_stride == 0)

    return summarize(ctx, run.acc, run.step_count, time.time() - run.clock.start_time)
