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

from training.cloud.session_deadline import SessionDeadlineReached
from training.data.resumable_sampler import resume_start_batch
from training.hardware.device_memory import is_out_of_memory, release_device_memory
from training.training.amp import GradClip, get_autocast_context, safe_backward
from training.training.context import TrainingContext
from training.training.step_progress import (
    BatchClock,
    batch_progress_line,
    deadline_reached,
    is_main_process,
    progress_payload,
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
    """One training epoch over ``ctx.data.train_loader``."""

    def __init__(self, ctx: TrainingContext, epoch: int) -> None:
        self.ctx = ctx
        self.epoch = epoch
        self.autocast_ctx = get_autocast_context(
            ctx.hardware.policy, ctx.hardware.device_info.device
        )
        self.totals = _TrainTotals()
        start_time = time.time()
        ctx.models.optimizer.zero_grad(set_to_none=True)
        num_batches = len(ctx.data.train_loader) if hasattr(ctx.data.train_loader, "__len__") else 0
        self.clock = BatchClock(num_batches, start_time, start_time)

    def resume(self) -> None:
        """Restore the totals of an interrupted epoch, if any."""
        ctx = self.ctx
        resume_step = resume_iteration(
            ctx.state.resume_state, "train", ctx.state.guards.train_step_handled
        )
        if resume_step <= 0:
            return
        self.clock.resume_step = resume_step
        ctx.state.guards.train_step_handled = True
        print(
            f"[RESUME] Fast-forwarding {resume_step} completed batches in Epoch {self.epoch}...",
            flush=True,
        )
        saved = getattr(ctx.state.resume_state, "train_accumulators", {})
        if saved:
            self.totals.total_loss = float(saved.get("total_loss", 0.0))
            self.totals.total_grad_norm = float(saved.get("total_grad_norm", 0.0))
            self.totals.step_count = int(saved.get("step_count", resume_step))
        else:
            self.totals.step_count = resume_step

    def forward(self, inputs: Any, targets: Any, task_idx: Any) -> Any:
        """Return the unscaled loss of one batch."""
        ctx = self.ctx
        train_step_fn = getattr(ctx.models.model, "train_step", None)
        if callable(train_step_fn):
            loss_dict = train_step_fn(inputs)
            return loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
        preds = ctx.models.model(inputs)
        if task_idx is not None:
            return ctx.models.criterion(preds, targets, task_idx)
        return ctx.models.criterion(preds, targets)

    def is_boundary(self, batch_idx: int, accumulated: int) -> bool:
        """True when ``accumulated`` micro-batches complete a step, or on the last batch."""
        ctx = self.ctx
        return accumulated % ctx.data.accumulation_steps == 0 or (batch_idx + 1) == len(
            ctx.data.train_loader
        )

    def step(self, batch: tuple[Any, Any, Any], batch_idx: int) -> tuple[Any, float]:
        """Forward and backward one unpacked batch; return ``(raw loss, gradient norm)``."""
        ctx = self.ctx
        with self.autocast_ctx:
            raw_loss = self.forward(*batch)
            scaled_loss = raw_loss / max(1, ctx.data.accumulation_steps)

        is_update_step = self.is_boundary(batch_idx, self.totals.accumulated_steps + 1)

        # In multi-GPU DDP, suppress all-reduce on micro-batches before the update step boundary
        bwd_ctx: Any = nullcontext()
        if (
            not is_update_step
            and ctx.models.parallel_strategy is not None
            and hasattr(ctx.models.parallel_strategy, "backward_context")
        ):
            bwd_ctx = ctx.models.parallel_strategy.backward_context()

        # Scale loss, compute backward, and collect gradient norm at step boundary
        with bwd_ctx:
            grad_norm = safe_backward(
                loss=scaled_loss,
                optimizer=ctx.models.optimizer,
                scaler=ctx.models.scaler,
                clip=GradClip(model=ctx.models.model, max_norm=1.0, should_unscale=is_update_step),
            )
        return raw_loss, grad_norm

    def relieve_oom(self, batch_idx: int) -> None:
        """Drop the gradients and let the governor shrink the batch (or flush the cache)."""
        ctx = self.ctx
        if ctx.models.optimizer is not None:
            ctx.models.optimizer.zero_grad(set_to_none=True)
        if hasattr(ctx.governance.governor, "handle_oom"):
            curr_b = getattr(ctx.data.train_loader, "batch_size", 16) or 16
            _, new_acc = ctx.governance.governor.handle_oom(current_batch=curr_b)
            ctx.data.accumulation_steps = new_acc
        else:
            release_device_memory("train OOM relief")
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
        if ctx.models.scaler is not None:
            ctx.models.scaler.step(ctx.models.optimizer)
            ctx.models.scaler.update()
        else:
            ctx.models.optimizer.step()
        ctx.models.optimizer.zero_grad(set_to_none=True)

    def save_progress(self, batch_idx: int, force: bool = False) -> None:
        """Offer an intra-epoch progress checkpoint to the lifecycle manager (``force``: save)."""
        ctx = self.ctx
        if ctx.governance.lifecycle_manager is None:
            return
        raw_m = ctx.models.raw_model if ctx.models.raw_model is not None else ctx.models.model
        manager = ctx.governance.lifecycle_manager
        save = manager.save_progress_now if force else manager.check_and_save_progress
        save(
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
        cur_lr = float(self.ctx.models.optimizer.param_groups[0]["lr"])
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
        current_lr = float(ctx.models.optimizer.param_groups[0]["lr"])
        if (
            ctx.models.scheduler is not None
            and not hasattr(ctx.models.scheduler, "step_batch")
            and not (ctx.governance.swa is not None and ctx.governance.swa.owns_lr(self.epoch))
        ):
            try:
                ctx.models.scheduler.step()
            except (TypeError, ValueError, RuntimeError) as exc:
                # e.g. ReduceLROnPlateau requires a metrics argument; the LR is left unchanged.
                logger.warning(
                    "Epoch scheduler step skipped for %s: %s",
                    type(ctx.models.scheduler).__name__,
                    exc,
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
    ctx.models.model.train()
    run = _TrainingPass(ctx, epoch)
    run.resume()

    first_batch = resume_start_batch(ctx.data.train_loader, run.clock.resume_step)
    for batch_idx, batch in enumerate(ctx.data.train_loader, start=first_batch):
        if run.clock.skip_resumed(batch_idx, "BATCH", "training"):
            continue
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Training cancelled by user request.")

        unpacked = _unpack_batch(batch, ctx.hardware.device_info.device)
        try:
            raw_loss, grad_norm = run.step(unpacked, batch_idx)
        except (torch.OutOfMemoryError, RuntimeError) as oom_err:
            if not is_out_of_memory(oom_err):
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
        if deadline_reached(ctx, batch_idx + 1):
            run.save_progress(batch_idx, force=True)
            raise SessionDeadlineReached(epoch, "train", batch_idx + 1)

    return run.finish()
