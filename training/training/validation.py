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
from training.training.val_metrics import BATCH_METRICS, ValBatch
from training.training.val_summary import summarize

logger = logging.getLogger("lemtrain.training.validation")


def _log_due(clock: BatchClock, step: int, now: float) -> bool:
    """True on warm-up steps, about every 1% of the pass, the last step, or after an 8 s gap."""
    log_interval = max(1, clock.num_batches // 100) if clock.num_batches > 100 else 10
    is_warmup = step in (1, 2, 5, 10)
    is_periodic = (step % log_interval == 0) or (step == clock.num_batches)
    is_heartbeat = now - clock.last_log_time >= 8.0 and step > 10
    return is_warmup or is_periodic or is_heartbeat


class _ValidationPass:
    """One validation pass over ``ctx.val_loader``."""

    def __init__(self, ctx: TrainingContext, epoch: int) -> None:
        self.ctx = ctx
        self.epoch = epoch
        self.raw_model = ctx.raw_model if ctx.raw_model is not None else ctx.model
        self.autocast_ctx = get_autocast_context(ctx.policy, ctx.device_info.device)
        self.acc = ValAccumulators()
        self.step_count = 0
        start_time = time.time()
        num_batches = (
            len(ctx.val_loader)
            if ctx.val_loader is not None and hasattr(ctx.val_loader, "__len__")
            else 0
        )
        self.clock = BatchClock(num_batches, start_time, time.time())

    def resume(self) -> None:
        """Restore accumulators of an interrupted validation pass, if any."""
        resume_step = resume_iteration(
            self.ctx.resume_state, "val", self.ctx.resumed_val_step_handled
        )
        if resume_step <= 0:
            return
        self.clock.resume_step = resume_step
        self.ctx.resumed_val_step_handled = True
        print(
            f"[RESUME] Fast-forwarding {resume_step} completed validation batches in Epoch "
            f"{self.epoch}...",
            flush=True,
        )
        saved = getattr(self.ctx.resume_state, "val_accumulators", {})
        if saved:
            self.acc.restore(saved)
            self.step_count = resume_step

    def forward(self, inputs: Any, targets: Any, task_idx: Any) -> tuple[Any, Any]:
        """Return ``(preds, loss)`` for one batch under autocast."""
        ctx = self.ctx
        with self.autocast_ctx:
            eval_step_fn = getattr(ctx.model, "eval_step", None)
            if callable(eval_step_fn):
                val_out = eval_step_fn(inputs)
                if isinstance(val_out, dict):
                    return val_out.get("preds", inputs), val_out["loss"]
                return inputs, val_out
            preds = ctx.model(inputs)
            if task_idx is not None:
                return preds, ctx.criterion(preds, targets, task_idx)
            return preds, ctx.criterion(preds, targets)

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
        if hasattr(self.ctx.governor, "handle_oom"):
            self.ctx.governor.handle_oom()
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
        if ctx.lifecycle_manager is None:
            return
        total_val = (
            len(ctx.val_loader)
            if ctx.val_loader is not None and hasattr(ctx.val_loader, "__len__")
            else 0
        )
        ctx.lifecycle_manager.check_and_save_progress(
            epoch=self.epoch,
            phase="val",
            step=self.step_count,
            total_steps=total_val,
            payload_builder=partial(
                _val_payload, ctx, self.epoch, self.raw_model, self.acc.snapshot()
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

    def batch_view(self, preds: Any, targets: Any) -> ValBatch:
        """Bundle one batch with the evaluation settings of the model."""
        ctx = self.ctx
        return ValBatch(
            preds=preds,
            targets=targets,
            sota_cfg=ctx.model_info.get("sota_targets", {}),
            dataset_type=ctx.model_info.get("dataset_type"),
            device=ctx.device_info.device,
            governor=ctx.governor,
        )


def _val_payload(
    ctx: TrainingContext, epoch: int, model: torch.nn.Module, snapshot: dict[str, Any]
) -> dict[str, Any]:
    return progress_payload(
        ctx,
        epoch,
        model,
        {
            "train_metrics": getattr(ctx, "last_train_metrics", {}),
            "val_accumulators": snapshot,
        },
    )


def validate_one_epoch(
    ctx: TrainingContext,
    epoch: int,
    cancel_check: Any = None,
) -> dict[str, float]:
    """Execute validation over the evaluation loader.

    Args:
        ctx: Training context holding model, validation data loader, and policies.
        epoch: Epoch index being evaluated.
        cancel_check: Optional callable returning True when cancellation is requested.

    Returns:
        dict[str, float]: Dictionary of validation metrics.
    """
    if ctx.val_loader is None or len(ctx.val_loader) == 0:
        return {"val_loss": 0.0}

    ctx.model.eval()
    run = _ValidationPass(ctx, epoch)
    run.resume()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.val_loader):
            if run.clock.skip_resumed(batch_idx, "VAL BATCH", "validation"):
                continue
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Validation cancelled by user request.")

            inputs, targets, task_idx = _unpack_batch(batch, ctx.device_info.device)
            outcome = run.guarded_forward(batch_idx, (inputs, targets, task_idx))
            if outcome is None:
                continue
            preds, loss = outcome
            run.acc.total_loss += float(loss.item())
            run.step_count += 1
            run.save_progress()
            run.log(loss)

            view = run.batch_view(preds, targets)
            for update in BATCH_METRICS:
                update(run.acc, view)

    return summarize(ctx, run.acc, run.step_count, time.time() - run.clock.start_time)
