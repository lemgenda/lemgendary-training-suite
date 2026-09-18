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


def train_one_epoch(ctx: TrainingContext, epoch: int) -> dict[str, float]:
    """Execute one training epoch across all batches in the data loader.

    Args:
        ctx: Training context holding model, optimizer, data loaders, and policies.
        epoch: Zero-indexed or one-indexed epoch number.

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

    for batch_idx, batch in enumerate(ctx.train_loader):
        inputs, targets, task_idx = _unpack_batch(batch, device)

        with autocast_ctx:
            if hasattr(ctx.model, "train_step"):
                loss_dict = ctx.model.train_step(inputs)
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
