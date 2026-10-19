"""Validation Engine for LemGendary Training Suite.

Executes deterministic evaluation passes under torch.no_grad(), computing task-specific
loss metrics, PSNR/SSIM reconstruction fidelity, or classification scores.
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import torch

from training.training.amp import compute_ssim_gpu, get_autocast_context
from training.training.context import TrainingContext
from training.training.epoch import _unpack_batch


def _compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio from Mean Squared Error."""
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def _extract_forex_tensors(
    preds: Any,
    targets: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract directional logits, magnitude predictions, and targets."""
    dir_logits: torch.Tensor | None = None
    mag_preds: torch.Tensor | None = None
    dir_targets: torch.Tensor | None = None
    mag_targets: torch.Tensor | None = None

    if isinstance(preds, dict):
        dir_logits = preds.get("direction_logits", preds.get("direction"))
        mag_preds = preds.get("magnitude")
    elif isinstance(preds, (tuple, list)) and len(preds) >= 2:
        dir_logits = preds[0] if isinstance(preds[0], torch.Tensor) else None
        mag_preds = preds[1] if isinstance(preds[1], torch.Tensor) else None

    if isinstance(targets, dict):
        dir_targets = targets.get("direction")
        mag_targets = targets.get("magnitude")
    elif isinstance(targets, (tuple, list)) and len(targets) >= 2:
        dir_targets = targets[0] if isinstance(targets[0], torch.Tensor) else None
        mag_targets = targets[1] if isinstance(targets[1], torch.Tensor) else None
    elif isinstance(targets, torch.Tensor) and targets.ndim == 1:
        dir_targets = targets

    return dir_logits, mag_preds, dir_targets, mag_targets


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
    has_forex_metrics = False
    total_forex_samples = 0
    correct_forex_samples = 0
    total_trades = 0
    winning_trades = 0
    total_forex_entropy = 0.0
    total_tp_mae = 0.0
    total_sl_mae = 0.0
    total_mag_samples = 0
    trade_returns: list[float] = []
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
                        "train_metrics": getattr(ctx, "last_train_metrics", {}),
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

            # Check for Forex / Directional predictions and targets
            dir_logits, mag_preds, dir_targets, mag_targets = _extract_forex_tensors(preds, targets)
            if dir_logits is not None and dir_targets is not None:
                has_forex_metrics = True
                pred_dir = torch.argmax(dir_logits, dim=-1)
                bs = dir_targets.size(0)
                total_forex_samples += bs
                correct_forex_samples += int((pred_dir == dir_targets).sum().item())

                # Directional trades: non-HOLD predictions (0 = SELL, 2 = BUY)
                directional_mask = (pred_dir != 1)
                total_trades += int(directional_mask.sum().item())
                winning_trades += int(((pred_dir == dir_targets) & directional_mask).sum().item())

                # Direction entropy
                probs = torch.softmax(dir_logits, dim=-1)
                batch_entropy = float(-(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item())
                total_forex_entropy += batch_entropy * bs

                # Magnitude MAE
                if mag_preds is not None and mag_targets is not None:
                    clamped_mag = torch.clamp(mag_preds, max=100.0)
                    if (
                        clamped_mag.ndim >= 2
                        and mag_targets.ndim >= 2
                        and clamped_mag.shape[-1] >= 2
                        and mag_targets.shape[-1] >= 2
                    ):
                        tp_err = torch.abs(clamped_mag[..., 0] - mag_targets[..., 0]).sum().item()
                        sl_err = torch.abs(clamped_mag[..., 1] - mag_targets[..., 1]).sum().item()
                        total_tp_mae += float(tp_err)
                        total_sl_mae += float(sl_err)
                        total_mag_samples += bs

                # Simulated PnL returns for Sharpe / Sortino / Profit Factor / MaxDD
                if directional_mask.any():
                    dirs_np = pred_dir[directional_mask].detach().cpu().numpy()
                    targets_np = dir_targets[directional_mask].detach().cpu().numpy()
                    mags_np = (
                        mag_targets[directional_mask].detach().cpu().numpy()
                        if mag_targets is not None and mag_targets.ndim >= 2
                        else None
                    )

                    for idx in range(len(dirs_np)):
                        p_d = int(dirs_np[idx])
                        t_d = int(targets_np[idx])
                        tp_val = float(mags_np[idx, 0]) if mags_np is not None else 15.0
                        sl_val = float(mags_np[idx, 1]) if mags_np is not None else 15.0

                        if p_d == t_d:
                            trade_returns.append(tp_val)
                        else:
                            trade_returns.append(-sl_val)

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

    if has_forex_metrics and total_forex_samples > 0:
        dir_acc = (correct_forex_samples / max(1, total_forex_samples)) * 100.0
        win_rate = (winning_trades / max(1, total_trades)) * 100.0 if total_trades > 0 else dir_acc
        dir_entropy = total_forex_entropy / max(1, total_forex_samples)
        tp_mae = total_tp_mae / max(1, total_mag_samples) if total_mag_samples > 0 else 0.0
        sl_mae = total_sl_mae / max(1, total_mag_samples) if total_mag_samples > 0 else 0.0

        if len(trade_returns) > 0:
            returns_arr = np.array(trade_returns, dtype=np.float64)
            pos_returns = returns_arr[returns_arr > 0]
            neg_returns = np.abs(returns_arr[returns_arr < 0])

            gross_profit = float(pos_returns.sum()) if len(pos_returns) > 0 else 0.0
            gross_loss = float(neg_returns.sum()) if len(neg_returns) > 0 else 0.0
            pf = gross_profit / max(1e-4, gross_loss) if gross_loss > 0 else (min(3.0, gross_profit) if gross_profit > 0 else 1.0)
            profit_factor = float(np.clip(pf, 0.0, 10.0))

            mean_r = float(np.mean(returns_arr))
            std_r = float(np.std(returns_arr))
            sharpe = (mean_r / max(1e-6, std_r)) * np.sqrt(min(252, len(returns_arr)))
            sharpe_ratio = float(np.clip(sharpe, -5.0, 10.0))

            downside = returns_arr[returns_arr < 0]
            downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-4
            sortino = (mean_r / max(1e-6, downside_std)) * np.sqrt(min(252, len(returns_arr)))
            sortino_ratio = float(np.clip(sortino, -5.0, 10.0))

            cum_r = np.cumsum(returns_arr)
            running_max = np.maximum.accumulate(cum_r)
            drawdown = running_max - cum_r
            max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            max_drawdown = float(np.clip(max_dd, 0.0, 100.0))
        else:
            profit_factor = 1.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0

        forex_quality = (
            (dir_acc * 40.0)
            + (win_rate * 30.0)
            + (min(profit_factor, 3.0) * 10.0)
            + (min(sharpe_ratio, 3.0) * 10.0)
            - (max_drawdown * 20.0)
        )

        results.update({
            "dir_acc": dir_acc,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "tp_mae": tp_mae,
            "sl_mae": sl_mae,
            "dir_entropy": dir_entropy,
            "quality_score": max(0.0, float(forex_quality)),
        })

    return results
