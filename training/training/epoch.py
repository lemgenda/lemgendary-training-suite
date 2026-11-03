"""Epoch Execution Engine for LemGendary Training Suite.

Handles one full training epoch over the dataset loader, including batch unpacking,
forward passes under AMP autocast, gradient accumulation, and telemetry collection.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from training.training.amp import get_autocast_context, safe_backward
from training.training.context import TrainingContext
from training.training.step_progress import (
    BatchClock,
    batch_progress_line,
    is_main_process,
    is_oom_error,
    progress_payload,
    release_cuda_cache,
    resume_iteration,
)
from training.utils.tensors import to_device

logger = logging.getLogger("lemtrain.training.epoch")


def _unpack_batch(batch: Any, device: torch.device) -> tuple[Any, Any, Any]:
    """Unpack batch into inputs, targets, and optional task indicators."""
    if isinstance(batch, (tuple, list)):
        batch_seq = list(batch)
        if len(batch_seq) >= 3:
            inputs, targets, task_idx = batch_seq[0], batch_seq[1], batch_seq[2]
        elif len(batch_seq) == 2:
            inputs, targets = batch_seq[0], batch_seq[1]
            task_idx = None
        elif len(batch_seq) == 1:
            inputs = batch_seq[0]
            targets = batch_seq[0]
            task_idx = None
        else:
            inputs, targets, task_idx = None, None, None
    elif isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("image", batch.get("features", batch)))
        targets = batch.get("targets", batch.get("target", batch.get("label", inputs)))
        task_idx = batch.get("task_idx", None)
    else:
        inputs = batch
        targets = batch
        task_idx = None

    inputs = to_device(inputs, device)
    targets = to_device(targets, device)
    if isinstance(task_idx, torch.Tensor):
        task_idx = task_idx.to(device, non_blocking=True)

    return inputs, targets, task_idx


@dataclass
class _TrainTotals:
    """Running totals of one training epoch."""

    total_loss: float = 0.0
    total_grad_norm: float = 0.0
    step_count: int = 0
    accumulated_steps: int = 0

    def snapshot(self) -> dict[str, Any]:
        """Progress-checkpoint snapshot (``train_accumulators``)."""
        return {
            "total_loss": self.total_loss,
            "total_grad_norm": self.total_grad_norm,
            "step_count": self.step_count,
        }


def _log_due(clock: BatchClock, batch_num: int, session_step: int, now: float) -> bool:
    """True on warm-up batches, periodic batches, the last batch, or after an 8 s gap."""
    log_interval = max(1, min(25, clock.num_batches // 50)) if clock.num_batches > 0 else 10
    is_warmup = batch_num in (1, 2, 5, 10) or session_step in (1, 2, 5, 10)
    is_periodic = batch_num % log_interval == 0 or batch_num == clock.num_batches
    is_heartbeat = now - clock.last_log_time >= 8.0 and batch_num > 10
    return is_warmup or is_periodic or is_heartbeat


class _TrainingPass:
    """One training epoch over ``ctx.train_loader``."""

    def __init__(self, ctx: TrainingContext, epoch: int) -> None:
        self.ctx = ctx
        self.epoch = epoch
        self.autocast_ctx = get_autocast_context(ctx.policy, ctx.device_info.device)
        self.totals = _TrainTotals()
        start_time = time.time()
        ctx.optimizer.zero_grad(set_to_none=True)
        num_batches = len(ctx.train_loader) if hasattr(ctx.train_loader, "__len__") else 0
        self.clock = BatchClock(num_batches, start_time, start_time)

    def resume(self) -> None:
        """Restore the totals of an interrupted epoch, if any."""
        ctx = self.ctx
        resume_step = resume_iteration(ctx.resume_state, "train", ctx.resumed_train_step_handled)
        if resume_step <= 0:
            return
        self.clock.resume_step = resume_step
        ctx.resumed_train_step_handled = True
        print(
            f"[RESUME] Fast-forwarding {resume_step} completed batches in Epoch {self.epoch}...",
            flush=True,
        )
        saved = getattr(ctx.resume_state, "train_accumulators", {})
        if saved:
            self.totals.total_loss = float(saved.get("total_loss", 0.0))
            self.totals.total_grad_norm = float(saved.get("total_grad_norm", 0.0))
            self.totals.step_count = int(saved.get("step_count", resume_step))
        else:
            self.totals.step_count = resume_step

    def forward(self, inputs: Any, targets: Any, task_idx: Any) -> Any:
        """Return the unscaled loss of one batch."""
        ctx = self.ctx
        train_step_fn = getattr(ctx.model, "train_step", None)
        if callable(train_step_fn):
            loss_dict = train_step_fn(inputs)
            return loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
        preds = ctx.model(inputs)
        if task_idx is not None:
            return ctx.criterion(preds, targets, task_idx)
        return ctx.criterion(preds, targets)

    def is_boundary(self, batch_idx: int, accumulated: int) -> bool:
        """True when ``accumulated`` micro-batches complete a step, or on the last batch."""
        ctx = self.ctx
        return accumulated % ctx.accumulation_steps == 0 or (batch_idx + 1) == len(ctx.train_loader)

    def step(self, batch: tuple[Any, Any, Any], batch_idx: int) -> tuple[Any, float]:
        """Forward and backward one unpacked batch; return ``(raw loss, gradient norm)``."""
        ctx = self.ctx
        with self.autocast_ctx:
            raw_loss = self.forward(*batch)
            scaled_loss = raw_loss / max(1, ctx.accumulation_steps)

        is_update_step = self.is_boundary(batch_idx, self.totals.accumulated_steps + 1)

        # In multi-GPU DDP, suppress all-reduce on micro-batches before the update step boundary
        bwd_ctx: Any = nullcontext()
        if (
            not is_update_step
            and ctx.parallel_strategy is not None
            and hasattr(ctx.parallel_strategy, "backward_context")
        ):
            bwd_ctx = ctx.parallel_strategy.backward_context()

        # Scale loss, compute backward, and collect gradient norm at step boundary
        with bwd_ctx:
            grad_norm = safe_backward(
                loss=scaled_loss,
                optimizer=ctx.optimizer,
                scaler=ctx.scaler,
                max_norm=1.0,
                model=ctx.model,
                should_unscale=is_update_step,
            )
        return raw_loss, grad_norm

    def relieve_oom(self, batch_idx: int) -> None:
        """Drop the gradients and let the governor shrink the batch (or flush the cache)."""
        ctx = self.ctx
        if ctx.optimizer is not None:
            ctx.optimizer.zero_grad(set_to_none=True)
        if hasattr(ctx.governor, "handle_oom"):
            curr_b = getattr(ctx.train_loader, "batch_size", 16) or 16
            _, new_acc = ctx.governor.handle_oom(current_batch=curr_b)
            ctx.accumulation_steps = new_acc
        else:
            release_cuda_cache()
        print(
            f"\n  [TRAIN WARNING] VRAM OOM intercepted on batch {batch_idx + 1}. Governor "
            "active mitigation applied.",
            flush=True,
        )

    def apply_update(self, batch_idx: int) -> None:
        """Step the optimizer (through the AMP scaler when present) at a step boundary."""
        ctx = self.ctx
        if not self.is_boundary(batch_idx, self.totals.accumulated_steps):
            return
        if ctx.scaler is not None:
            ctx.scaler.step(ctx.optimizer)
            ctx.scaler.update()
        else:
            ctx.optimizer.step()
        ctx.optimizer.zero_grad(set_to_none=True)

    def save_progress(self, batch_idx: int) -> None:
        """Offer an intra-epoch progress checkpoint to the lifecycle manager."""
        ctx = self.ctx
        if ctx.lifecycle_manager is None:
            return
        raw_m = ctx.raw_model if ctx.raw_model is not None else ctx.model
        ctx.lifecycle_manager.check_and_save_progress(
            epoch=self.epoch,
            phase="train",
            step=batch_idx + 1,
            total_steps=self.clock.num_batches,
            payload_builder=partial(
                progress_payload,
                ctx,
                self.epoch,
                raw_m,
                {"train_accumulators": self.totals.snapshot()},
            ),
        )

    def log(self, batch_idx: int, raw_loss: Any) -> None:
        """Print the batch telemetry line when due (rank 0 only)."""
        clock = self.clock
        batch_num = batch_idx + 1
        now = time.time()
        session_step = self.totals.step_count - clock.resume_step
        if not (is_main_process() and _log_due(clock, batch_num, session_step, now)):
            return
        clock.last_log_time = now
        cur_lr = float(self.ctx.optimizer.param_groups[0]["lr"])
        rate, elapsed_batch = clock.rate(batch_num, now)
        metrics = (
            f"Loss: {float(raw_loss.item()):.4f} | "
            f"AvgLoss: {self.totals.total_loss / max(1, self.totals.step_count):.4f} | "
            f"LR: {cur_lr:.6f}"
        )
        line = batch_progress_line(
            "BATCH", (batch_num, clock.num_batches), metrics, rate, elapsed_batch
        )
        print(line, end="\n" if batch_num == clock.num_batches else "", flush=True)

    def finish(self) -> dict[str, float]:
        """Step an epoch-based scheduler and return the epoch telemetry."""
        ctx = self.ctx
        if is_main_process():
            print(flush=True)
        elapsed = time.time() - self.clock.start_time
        steps = max(1, self.totals.step_count)
        current_lr = float(ctx.optimizer.param_groups[0]["lr"])
        if (
            ctx.scheduler is not None
            and not hasattr(ctx.scheduler, "step_batch")
            and not (ctx.swa is not None and ctx.swa.owns_lr(self.epoch))
        ):
            try:
                ctx.scheduler.step()
            except (TypeError, ValueError, RuntimeError) as exc:
                # e.g. ReduceLROnPlateau requires a metrics argument; the LR is left unchanged.
                logger.warning(
                    "Epoch scheduler step skipped for %s: %s", type(ctx.scheduler).__name__, exc
                )
        return {
            "train_loss": self.totals.total_loss / steps,
            "grad_norm": self.totals.total_grad_norm / steps,
            "lr": current_lr,
            "epoch_time": elapsed,
        }


def train_one_epoch(
    ctx: TrainingContext,
    epoch: int,
    cancel_check: Any = None,
) -> dict[str, float]:
    """Execute one training epoch across all batches in the data loader.

    Args:
        ctx: Training context holding model, optimizer, data loaders, and policies.
        epoch: Zero-indexed or one-indexed epoch number.
        cancel_check: Optional callable returning True when cancellation is requested.

    Returns:
        dict[str, float]: Telemetry dictionary containing loss, lr, grad_norm, and elapsed time.
    """
    ctx.model.train()
    run = _TrainingPass(ctx, epoch)
    run.resume()

    for batch_idx, batch in enumerate(ctx.train_loader):
        if run.clock.skip_resumed(batch_idx, "BATCH", "training"):
            continue
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Training cancelled by user request.")

        unpacked = _unpack_batch(batch, ctx.device_info.device)
        try:
            raw_loss, grad_norm = run.step(unpacked, batch_idx)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
            if not is_oom_error(oom_err):
                raise
            run.relieve_oom(batch_idx)
            continue

        totals = run.totals
        totals.total_loss += float(raw_loss.item())
        totals.total_grad_norm += grad_norm
        totals.step_count += 1
        totals.accumulated_steps += 1
        run.apply_update(batch_idx)

        # Periodic cache clearing to prevent virtual address fragmentation on <=16GB GPUs
        if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        run.save_progress(batch_idx)
        run.log(batch_idx, raw_loss)

    return run.finish()
