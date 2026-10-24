"""Epoch Execution Engine for LemGendary Training Suite.

Handles one full training epoch over the dataset loader, including batch unpacking,
forward passes under AMP autocast, gradient accumulation, and telemetry collection.
"""

from __future__ import annotations

import time
from typing import Any
import torch

from training.training.amp import get_autocast_context, safe_backward
from training.training.context import TrainingContext


def _unpack_batch(batch: Any, device: torch.device) -> tuple[Any, Any, Any]:
    """Unpack batch into inputs, targets, and optional task indicators."""
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            inputs, targets, task_idx = batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            inputs, targets = batch[0], batch[1]
            task_idx = None
        else:
            inputs = batch[0]
            targets = batch[0]
            task_idx = None
    elif isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("image", batch.get("features", batch)))
        targets = batch.get("targets", batch.get("target", batch.get("label", inputs)))
        task_idx = batch.get("task_idx", None)
    else:
        inputs = batch
        targets = batch
        task_idx = None

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device, non_blocking=True)
    elif isinstance(inputs, dict):
        inputs = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    if isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    elif isinstance(targets, dict):
        targets = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

    if isinstance(task_idx, torch.Tensor):
        task_idx = task_idx.to(device, non_blocking=True)

    return inputs, targets, task_idx


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
    device = ctx.device_info.device
    autocast_ctx = get_autocast_context(ctx.policy, device)

    total_loss = 0.0
    total_grad_norm = 0.0
    step_count = 0
    accumulated_steps = 0
    start_time = time.time()

    ctx.optimizer.zero_grad(set_to_none=True)

    num_batches = len(ctx.train_loader) if hasattr(ctx.train_loader, "__len__") else 0
    log_interval = max(1, min(25, num_batches // 50)) if num_batches > 0 else 10
    last_log_time = start_time

    resume_step = 0
    if (
        ctx.resume_state is not None
        and ctx.resume_state.is_intra_epoch
        and ctx.resume_state.phase == "train"
        and ctx.resume_state.iteration > 0
        and not getattr(ctx, "_resumed_train_step_handled", False)
    ):
        resume_step = ctx.resume_state.iteration
        setattr(ctx, "_resumed_train_step_handled", True)
        print(f"[RESUME] Fast-forwarding {resume_step} completed batches in Epoch {epoch}...", flush=True)

        acc = getattr(ctx.resume_state, "train_accumulators", {})
        if acc:
            total_loss = float(acc.get("total_loss", 0.0))
            total_grad_norm = float(acc.get("total_grad_norm", 0.0))
            step_count = int(acc.get("step_count", resume_step))
        else:
            step_count = resume_step

    ff_interval = max(1, min(500, resume_step // 20)) if resume_step > 0 else 100

    for batch_idx, batch in enumerate(ctx.train_loader):
        if batch_idx < resume_step:
            if (batch_idx + 1) % ff_interval == 0 or (batch_idx + 1) == resume_step:
                ff_pct = ((batch_idx + 1) / max(1, num_batches)) * 100.0
                print(
                    f"  [FAST-FORWARD] [BATCH {batch_idx + 1:>{len(str(num_batches))}}/{num_batches}] "
                    f"({ff_pct:.1f}%) Skipping processed training batches...",
                    flush=True,
                )
            continue

        if cancel_check is not None and cancel_check():
            raise InterruptedError("Training cancelled by user request.")

        inputs, targets, task_idx = _unpack_batch(batch, device)

        raw_loss: Any
        with autocast_ctx:
            train_step_fn = getattr(ctx.model, "train_step", None)
            if callable(train_step_fn):
                loss_dict = train_step_fn(inputs)
                raw_loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            elif task_idx is not None:
                preds = ctx.model(inputs)
                raw_loss = ctx.criterion(preds, targets, task_idx)
            else:
                preds = ctx.model(inputs)
                raw_loss = ctx.criterion(preds, targets)

            scaled_loss = raw_loss / max(1, ctx.accumulation_steps)

        # Scale loss, compute backward, and collect gradient norm
        grad_norm = safe_backward(
            loss=scaled_loss,
            optimizer=ctx.optimizer,
            scaler=ctx.scaler,
            max_norm=1.0,
            model=ctx.model,
        )

        total_loss += float(raw_loss.item())
        total_grad_norm += grad_norm
        step_count += 1
        accumulated_steps += 1

        if accumulated_steps % ctx.accumulation_steps == 0 or (batch_idx + 1) == len(ctx.train_loader):
            if ctx.scaler is not None:
                ctx.scaler.step(ctx.optimizer)
                ctx.scaler.update()
            else:
                ctx.optimizer.step()

            ctx.optimizer.zero_grad(set_to_none=True)

        # Intra-epoch progress check (15 min interval, strictly bounded to [5%, 50%])
        if ctx.lifecycle_manager is not None:
            raw_m = ctx.raw_model if ctx.raw_model is not None else ctx.model
            train_acc_snapshot = {
                "total_loss": total_loss,
                "total_grad_norm": total_grad_norm,
                "step_count": step_count,
            }
            ctx.lifecycle_manager.check_and_save_progress(
                epoch=epoch,
                phase="train",
                step=batch_idx + 1,
                total_steps=num_batches,
                payload_builder=lambda: {
                    "epoch": epoch,
                    "model_name": ctx.model_name,
                    "model_state": raw_m.state_dict(),
                    "optimizer_state": ctx.optimizer.state_dict(),
                    "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
                    "governor_state": ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None,
                    "train_accumulators": dict(train_acc_snapshot),
                },
            )

        now = time.time()
        is_warmup = (batch_idx + 1) in (1, 2, 5, 10) or (step_count - resume_step in (1, 2, 5, 10))
        time_elapsed_since_log = now - last_log_time
        is_periodic = ((batch_idx + 1) % log_interval == 0) or ((batch_idx + 1) == num_batches)
        is_heartbeat = (time_elapsed_since_log >= 8.0 and (batch_idx + 1) > 10)

        if is_warmup or is_periodic or is_heartbeat:
            last_log_time = now
            batch_loss = float(raw_loss.item())
            cur_lr = float(ctx.optimizer.param_groups[0]["lr"])
            elapsed_batch = now - start_time
            processed_in_session = max(1, (batch_idx + 1) - resume_step)
            rate = processed_in_session / max(0.001, elapsed_batch)
            rate_str = f"{rate:.2f} it/s" if rate >= 1.0 else f"{1.0 / max(1e-4, rate):.1f} s/it"

            if num_batches > 0:
                pct = ((batch_idx + 1) / num_batches) * 100.0
                remaining = max(0, num_batches - (batch_idx + 1))
                eta_s = int(remaining / max(1e-4, rate))
                eta_m, eta_sec = divmod(eta_s, 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = f"ETA {eta_h}h{eta_m:02d}m" if eta_h > 0 else f"ETA {eta_m:02d}m{eta_sec:02d}s"

                print(
                    f"  [BATCH {batch_idx + 1:>{len(str(num_batches))}}/{num_batches}] "
                    f"({pct:.2f}%) "
                    f"Loss: {batch_loss:.4f} | "
                    f"AvgLoss: {total_loss / max(1, step_count):.4f} | "
                    f"LR: {cur_lr:.6f} | "
                    f"{rate_str} | "
                    f"{eta_str} | "
                    f"{elapsed_batch:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"  [BATCH {batch_idx + 1}] "
                    f"Loss: {batch_loss:.4f} | "
                    f"AvgLoss: {total_loss / max(1, step_count):.4f} | "
                    f"LR: {cur_lr:.6f} | "
                    f"{rate_str} | "
                    f"{elapsed_batch:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, step_count)
    avg_grad_norm = total_grad_norm / max(1, step_count)

    # Step epoch-based scheduler
    current_lr = float(ctx.optimizer.param_groups[0]["lr"])
    if ctx.scheduler is not None and not hasattr(ctx.scheduler, "step_batch"):
        try:
            ctx.scheduler.step()
        except Exception:
            pass

    return {
        "train_loss": avg_loss,
        "grad_norm": avg_grad_norm,
        "lr": current_lr,
        "epoch_time": elapsed,
    }
