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
    device = ctx.device_info.device
    autocast_ctx = get_autocast_context(ctx.policy, device)

    total_loss = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    has_image_metrics = False
    step_count = 0
    start_time = time.time()
    num_val_batches = len(ctx.val_loader) if hasattr(ctx.val_loader, "__len__") else 0
    log_interval = max(1, num_val_batches // 100) if num_val_batches > 100 else 10
    last_log_time = time.time()

    with torch.no_grad():
        for batch in ctx.val_loader:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Validation cancelled by user request.")

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

            # Intra-epoch progress check during validation (15 min interval, strictly bounded to [5%, 50%])
            if ctx.lifecycle_manager is not None:
                raw_m = ctx.raw_model if ctx.raw_model is not None else ctx.model
                total_val = len(ctx.val_loader) if hasattr(ctx.val_loader, "__len__") else 0
                ctx.lifecycle_manager.check_and_save_progress(
                    epoch=epoch,
                    phase="val",
                    step=step_count,
                    total_steps=total_val,
                    payload_builder=lambda: {
                        "epoch": epoch,
                        "model_name": ctx.model_name,
                        "model_state": raw_m.state_dict(),
                        "optimizer_state": ctx.optimizer.state_dict(),
                        "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
                        "governor_state": ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None,
                    },
                )

            # Validation batch telemetry progress logging
            now = time.time()
            is_warmup = step_count in (1, 2, 5, 10)
            time_elapsed_since_log = now - last_log_time
            is_periodic = (step_count % log_interval == 0) or (step_count == num_val_batches)
            is_heartbeat = (time_elapsed_since_log >= 8.0 and step_count > 10)

            if is_warmup or is_periodic or is_heartbeat:
                last_log_time = now
                batch_loss = float(loss.item())
                elapsed_batch = now - start_time
                rate = step_count / max(0.001, elapsed_batch)
                rate_str = f"{rate:.2f} it/s" if rate >= 1.0 else f"{1.0 / max(1e-4, rate):.1f} s/it"

                if num_val_batches > 0:
                    pct = (step_count / num_val_batches) * 100.0
                    remaining = max(0, num_val_batches - step_count)
                    eta_s = int(remaining / max(1e-4, rate))
                    eta_m, eta_sec = divmod(eta_s, 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"ETA {eta_h}h{eta_m:02d}m" if eta_h > 0 else f"ETA {eta_m:02d}m{eta_sec:02d}s"

                    print(
                        f"  [VAL BATCH {step_count:>{len(str(num_val_batches))}}/{num_val_batches}] "
                        f"({pct:.2f}%) "
                        f"ValLoss: {batch_loss:.4f} | "
                        f"AvgValLoss: {total_loss / step_count:.4f} | "
                        f"{rate_str} | "
                        f"{eta_str} | "
                        f"{elapsed_batch:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"  [VAL BATCH {step_count}] "
                        f"ValLoss: {batch_loss:.4f} | "
                        f"AvgValLoss: {total_loss / step_count:.4f} | "
                        f"{rate_str} | "
                        f"{elapsed_batch:.1f}s",
                        flush=True,
                    )

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
