"""Validation Engine for LemGendary Training Suite.

Executes deterministic evaluation passes under torch.no_grad(), computing task-specific
loss metrics, PSNR/SSIM reconstruction fidelity, or classification scores.
"""

from __future__ import annotations

import math
import time
from typing import Any
import torch

from training.training.amp import compute_ssim_gpu, get_autocast_context
from training.training.context import TrainingContext
from training.training.epoch import _unpack_batch


def _compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio from Mean Squared Error."""
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def validate_one_epoch(ctx: TrainingContext, epoch: int) -> dict[str, float]:
    """Execute validation over the evaluation loader.

    Args:
        ctx: Training context holding model, validation data loader, and policies.
        epoch: Epoch index being evaluated.

    Returns:
        dict[str, float]: Dictionary of validation metrics.
    """
    if ctx.val_loader is None or len(ctx.val_loader) == 0:
        return {"val_loss": 0.0}

    ctx.model.eval()
    device = ctx.device_info.device
    autocast_ctx = get_autocast_context(ctx.policy, device)

    total_loss = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    has_image_metrics = False
    step_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in ctx.val_loader:
            inputs, targets, task_idx = _unpack_batch(batch, device)

            with autocast_ctx:
                if hasattr(ctx.model, "eval_step"):
                    val_out = ctx.model.eval_step(inputs)
                    loss = val_out["loss"] if isinstance(val_out, dict) else val_out
                    preds = val_out.get("preds", inputs) if isinstance(val_out, dict) else inputs
                elif task_idx is not None:
                    preds = ctx.model(inputs)
                    loss = ctx.criterion(preds, targets, task_idx)
                else:
                    preds = ctx.model(inputs)
                    loss = ctx.criterion(preds, targets)

            total_loss += float(loss.item())
            step_count += 1

            # Check if outputs and targets are 4D image tensors [B, C, H, W] for PSNR/SSIM
            if (
                isinstance(preds, torch.Tensor)
                and isinstance(targets, torch.Tensor)
                and preds.ndim == 4
                and targets.ndim == 4
                and preds.shape == targets.shape
            ):
                has_image_metrics = True
                mse = float(torch.nn.functional.mse_loss(preds, targets).item())
                total_mse += mse
                try:
                    ssim = float(compute_ssim_gpu(preds, targets).item())
                    total_ssim += ssim
                except Exception:
                    pass

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, step_count)
    results: dict[str, float] = {
        "val_loss": avg_loss,
        "val_time": elapsed,
    }

    if has_image_metrics and step_count > 0:
        avg_mse = total_mse / step_count
        results["psnr"] = _compute_psnr(avg_mse)
        results["ssim"] = total_ssim / step_count

    return results
