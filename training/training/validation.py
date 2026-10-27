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

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    LearnedPerceptualImagePatchSimilarity = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None

try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY_STATS = True
except ImportError:
    pearsonr = None
    spearmanr = None
    _HAS_SCIPY_STATS = False


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
    raw_m = ctx.raw_model if ctx.raw_model is not None else ctx.model
    device = ctx.device_info.device
    autocast_ctx = get_autocast_context(ctx.policy, device)

    total_loss = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_lpips = 0.0
    lpips_count = 0
    lpips_metric_module: Any = None
    fid_metric_module: Any = None
    fid_batches_accumulated = 0
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

    # Quality & Ranking accumulators (NIMA / AARD)
    quality_preds_list: list[float] = []
    quality_targets_list: list[float] = []
    total_rank_margin_loss = 0.0
    rank_margin_count = 0

    # Categorical Classification accumulators (NSFW / Authenticity)
    total_cls_correct = 0
    total_cls_samples = 0

    # Parameter Prediction / UPN MAE accumulators
    total_upn_mae = 0.0
    upn_mae_count = 0

    # Segmentation mIoU accumulators (ParseNet)
    total_miou_inter = 0.0
    total_miou_union = 0.0

    # Object & Face Detection accumulators (YOLO / RetinaFace)
    detection_ap_scores: list[float] = []

    step_count = 0
    start_time = time.time()
    num_val_batches = len(ctx.val_loader) if hasattr(ctx.val_loader, "__len__") else 0
    log_interval = max(1, num_val_batches // 100) if num_val_batches > 100 else 10
    last_log_time = time.time()

    resume_step = 0
    if (
        ctx.resume_state is not None
        and ctx.resume_state.is_intra_epoch
        and ctx.resume_state.phase == "val"
        and ctx.resume_state.iteration > 0
        and not getattr(ctx, "_resumed_val_step_handled", False)
    ):
        resume_step = ctx.resume_state.iteration
        ctx._resumed_val_step_handled = True
        print(f"[RESUME] Fast-forwarding {resume_step} completed validation batches in Epoch {epoch}...", flush=True)

        # Restore running accumulators if available
        acc = getattr(ctx.resume_state, "val_accumulators", {})
        if acc:
            total_loss = float(acc.get("total_loss", 0.0))
            total_ssim = float(acc.get("total_ssim", 0.0))
            total_mse = float(acc.get("total_mse", 0.0))
            total_lpips = float(acc.get("total_lpips", 0.0))
            lpips_count = int(acc.get("lpips_count", 0))
            fid_batches_accumulated = int(acc.get("fid_batches_accumulated", 0))
            has_image_metrics = bool(acc.get("has_image_metrics", False))
            has_forex_metrics = bool(acc.get("has_forex_metrics", False))
            total_forex_samples = int(acc.get("total_forex_samples", 0))
            correct_forex_samples = int(acc.get("correct_forex_samples", 0))
            total_trades = int(acc.get("total_trades", 0))
            winning_trades = int(acc.get("winning_trades", 0))
            total_forex_entropy = float(acc.get("total_forex_entropy", 0.0))
            total_tp_mae = float(acc.get("total_tp_mae", 0.0))
            total_sl_mae = float(acc.get("total_sl_mae", 0.0))
            total_mag_samples = int(acc.get("total_mag_samples", 0))
            trade_returns = list(acc.get("trade_returns", []))
            quality_preds_list = list(acc.get("quality_preds_list", []))
            quality_targets_list = list(acc.get("quality_targets_list", []))
            total_cls_correct = int(acc.get("total_cls_correct", 0))
            total_cls_samples = int(acc.get("total_cls_samples", 0))
            total_upn_mae = float(acc.get("total_upn_mae", 0.0))
            upn_mae_count = int(acc.get("upn_mae_count", 0))
            total_miou_inter = float(acc.get("total_miou_inter", 0.0))
            total_miou_union = float(acc.get("total_miou_union", 0.0))
            detection_ap_scores = list(acc.get("detection_ap_scores", []))
            step_count = resume_step

    ff_interval = max(1, min(500, resume_step // 20)) if resume_step > 0 else 100

    with torch.no_grad():
        for batch_idx, batch in enumerate(ctx.val_loader):
            if batch_idx < resume_step:
                if (batch_idx + 1) % ff_interval == 0 or (batch_idx + 1) == resume_step:
                    ff_pct = ((batch_idx + 1) / max(1, num_val_batches)) * 100.0
                    print(
                        f"  [FAST-FORWARD] [VAL BATCH {batch_idx + 1:>{len(str(num_val_batches))}}/{num_val_batches}] "
                        f"({ff_pct:.1f}%) Skipping processed validation batches...",
                        flush=True,
                    )
                continue

            if cancel_check is not None and cancel_check():
                raise InterruptedError("Validation cancelled by user request.")

            inputs, targets, task_idx = _unpack_batch(batch, device)

            loss: Any
            with autocast_ctx:
                eval_step_fn = getattr(ctx.model, "eval_step", None)
                if callable(eval_step_fn):
                    val_out = eval_step_fn(inputs)
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
                total_val = len(ctx.val_loader) if hasattr(ctx.val_loader, "__len__") else 0
                val_acc_snapshot = {
                    "total_loss": total_loss,
                    "total_ssim": total_ssim,
                    "total_mse": total_mse,
                    "total_lpips": total_lpips,
                    "lpips_count": lpips_count,
                    "fid_batches_accumulated": fid_batches_accumulated,
                    "has_image_metrics": has_image_metrics,
                    "has_forex_metrics": has_forex_metrics,
                    "total_forex_samples": total_forex_samples,
                    "correct_forex_samples": correct_forex_samples,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "total_forex_entropy": total_forex_entropy,
                    "total_tp_mae": total_tp_mae,
                    "total_sl_mae": total_sl_mae,
                    "total_mag_samples": total_mag_samples,
                    "trade_returns": list(trade_returns),
                    "quality_preds_list": list(quality_preds_list),
                    "quality_targets_list": list(quality_targets_list),
                    "total_cls_correct": total_cls_correct,
                    "total_cls_samples": total_cls_samples,
                    "total_upn_mae": total_upn_mae,
                    "upn_mae_count": upn_mae_count,
                    "total_miou_inter": total_miou_inter,
                    "total_miou_union": total_miou_union,
                    "detection_ap_scores": list(detection_ap_scores),
                }

                def _build_progress_payload(
                    target_model: torch.nn.Module = raw_m,
                    acc_dict: dict[str, Any] = val_acc_snapshot,
                ) -> dict[str, Any]:
                    return {
                        "epoch": epoch,
                        "model_name": ctx.model_name,
                        "model_state": target_model.state_dict(),
                        "optimizer_state": ctx.optimizer.state_dict(),
                        "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
                        "governor_state": ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None,
                        "train_metrics": getattr(ctx, "last_train_metrics", {}),
                        "val_accumulators": acc_dict,
                    }

                ctx.lifecycle_manager.check_and_save_progress(
                    epoch=epoch,
                    phase="val",
                    step=step_count,
                    total_steps=total_val,
                    payload_builder=_build_progress_payload,
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
                processed_in_session = max(1, step_count - resume_step)
                rate = processed_in_session / max(0.001, elapsed_batch)
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
            eval_pred = preds.get("restored_image", preds) if isinstance(preds, dict) else preds
            eval_target = targets.get("image", targets) if isinstance(targets, dict) else targets

            sota_cfg = ctx.model_info.get("sota_targets", {})

            # 1. Vision Restoration / Image Enhancement (PSNR, SSIM, LPIPS, FID)
            eval_pred = preds.get("restored_image", preds) if isinstance(preds, dict) else preds
            eval_target = targets.get("image", targets) if isinstance(targets, dict) else targets

            if (
                isinstance(eval_pred, torch.Tensor)
                and isinstance(eval_target, torch.Tensor)
                and eval_pred.ndim == 4
                and eval_target.ndim == 4
                and eval_pred.shape == eval_target.shape
            ):
                has_image_metrics = True
                mse = float(torch.nn.functional.mse_loss(eval_pred, eval_target).item())
                total_mse += mse
                try:
                    ssim = float(compute_ssim_gpu(eval_pred, eval_target).item())
                    total_ssim += ssim
                except Exception:
                    pass

                # LPIPS perceptual evaluation (strictly FP32 without autocast half-precision errors)
                if "lpips" in sota_cfg and LearnedPerceptualImagePatchSimilarity is not None:
                    if lpips_metric_module is None:
                        try:
                            lpips_metric_module = LearnedPerceptualImagePatchSimilarity(
                                net_type="vgg",
                                normalize=False,
                            ).to(device)
                            lpips_metric_module.eval()
                            for p_param in lpips_metric_module.parameters():
                                p_param.requires_grad = False
                        except Exception as exc:
                            print(f"  [VAL NOTICE] LPIPS metric module initialization failed: {exc}", flush=True)
                            lpips_metric_module = False

                    if lpips_metric_module is not None and lpips_metric_module is not False:
                        try:
                            p_scaled = (torch.clamp(eval_pred, 0.0, 1.0) * 2.0 - 1.0).float()
                            t_scaled = (torch.clamp(eval_target, 0.0, 1.0) * 2.0 - 1.0).float()
                            batch_lpips = float(lpips_metric_module(p_scaled, t_scaled).item())
                            total_lpips += batch_lpips
                            lpips_count += 1
                        except Exception as exc:
                            print(f"  [VAL NOTICE] LPIPS evaluation error: {exc}", flush=True)

                # Frechet Inception Distance (FID) feature accumulation
                if "fid" in sota_cfg and FrechetInceptionDistance is not None:
                    if fid_metric_module is None:
                        try:
                            fid_metric_module = FrechetInceptionDistance(
                                feature=64,
                                reset_real_features=True,
                            ).to(device)
                            fid_metric_module.eval()
                        except Exception as exc:
                            print(f"  [VAL NOTICE] FID metric module initialization failed: {exc}", flush=True)
                            fid_metric_module = False

                    if fid_metric_module is not None and fid_metric_module is not False:
                        try:
                            p_u8 = torch.clamp(eval_pred.detach() * 255.0, 0.0, 255.0).to(torch.uint8)
                            t_u8 = torch.clamp(eval_target.detach() * 255.0, 0.0, 255.0).to(torch.uint8)
                            if p_u8.shape[1] == 1:
                                p_u8 = p_u8.repeat(1, 3, 1, 1)
                                t_u8 = t_u8.repeat(1, 3, 1, 1)
                            fid_metric_module.update(t_u8, real=True)
                            fid_metric_module.update(p_u8, real=False)
                            fid_batches_accumulated += 1
                        except Exception as exc:
                            print(f"  [VAL NOTICE] FID update error: {exc}", flush=True)

            # 2. Quality Assessment & Aesthetic Ranking (NIMA: PLCC, SRCC, Rank Margin)
            is_quality_model = (
                any(k in sota_cfg for k in ("plcc", "srcc", "rank_margin"))
                or ctx.model_info.get("dataset_type") == "quality"
            )
            if is_quality_model and isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                if preds.ndim >= 2 and targets.ndim >= 2 and preds.shape[-1] >= 10 and targets.shape[-1] >= 10:
                    try:
                        p_probs = torch.softmax(preds.float(), dim=-1)
                        t_probs = targets.float() / torch.clamp(targets.float().sum(dim=-1, keepdim=True), min=1e-6)
                        rank_w = torch.arange(1, 11, dtype=torch.float32, device=preds.device)
                        p_mean = (p_probs * rank_w).sum(dim=-1).detach().cpu().numpy().tolist()
                        t_mean = (t_probs * rank_w).sum(dim=-1).detach().cpu().numpy().tolist()
                        quality_preds_list.extend(p_mean)
                        quality_targets_list.extend(t_mean)

                        # Rank margin loss tracking
                        if preds.size(0) > 1 and "rank_margin" in sota_cfg:
                            p_diff = preds.float()[:, 0] - preds.float()[:, 1] if preds.size(0) == 2 else None
                            p_m_t = (p_probs * rank_w).sum(dim=-1)
                            t_m_t = (t_probs * rank_w).sum(dim=-1)
                            p_diff = p_m_t.unsqueeze(0) - p_m_t.unsqueeze(1)
                            t_diff = t_m_t.unsqueeze(0) - t_m_t.unsqueeze(1)
                            t_sign = torch.sign(t_diff)
                            margin_val = float(sota_cfg.get("rank_margin", 0.05))
                            rank_loss = torch.relu(margin_val - t_sign * p_diff)
                            mask = (t_sign != 0).float()
                            if mask.sum() > 0:
                                total_rank_margin_loss += float((rank_loss * mask).sum().item() / mask.sum().item())
                                rank_margin_count += 1
                    except Exception as exc:
                        print(f"  [VAL NOTICE] Quality metrics evaluation error: {exc}", flush=True)

            # 3. Categorical Classification (universal_nsfw_classification, nima_authenticity)
            if "accuracy" in sota_cfg and not has_forex_metrics:
                try:
                    if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                        if preds.ndim >= 2 and (targets.ndim == 1 or targets.shape[-1] == 1):
                            pred_labels = torch.argmax(preds, dim=-1).squeeze()
                            tgt_labels = targets.squeeze().long()
                            total_cls_correct += (pred_labels == tgt_labels).sum().item()
                            total_cls_samples += tgt_labels.numel()
                        elif preds.ndim == 1 and targets.ndim == 1:
                            pred_bin = (torch.sigmoid(preds) > 0.5).long()
                            total_cls_correct += (pred_bin == targets.long()).sum().item()
                            total_cls_samples += targets.numel()
                except Exception as exc:
                    print(f"  [VAL NOTICE] Classification accuracy evaluation error: {exc}", flush=True)

            # 4. Parameter Prediction / UPN (upn_v2 MAE)
            if "mae" in sota_cfg and not has_forex_metrics:
                try:
                    if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                        upn_err = torch.abs(preds.float() - targets.float()).mean().item()
                        total_upn_mae += float(upn_err)
                        upn_mae_count += 1
                except Exception as exc:
                    print(f"  [VAL NOTICE] UPN MAE evaluation error: {exc}", flush=True)

            # 5. Semantic Segmentation (parsenet mIoU)
            if "miou" in sota_cfg:
                try:
                    if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                        pred_seg = torch.argmax(preds, dim=1) if preds.ndim == 4 else preds
                        tgt_seg = targets.squeeze(1) if targets.ndim == 4 else targets
                        inter = torch.logical_and(tgt_seg > 0, pred_seg == tgt_seg).sum().item()
                        union = torch.logical_or(tgt_seg > 0, pred_seg > 0).sum().item()
                        total_miou_inter += float(inter)
                        total_miou_union += float(union)
                except Exception as exc:
                    print(f"  [VAL NOTICE] Segmentation mIoU evaluation error: {exc}", flush=True)

            # 6. Object / Face Detection (retinaface, yolov8n)
            has_det_cfg = any(k in sota_cfg for k in ("map50", "map50_95", "map_easy", "map_medium", "map_hard"))
            if has_det_cfg:
                try:
                    if isinstance(preds, (tuple, list)) and len(preds) >= 2 and isinstance(targets, torch.Tensor):
                        p_conf = preds[1] if isinstance(preds[1], torch.Tensor) else preds[0]
                        t_conf = targets[:, 0:1] if targets.ndim >= 2 else targets
                        conf_diff = torch.abs(torch.sigmoid(p_conf.float()) - t_conf.float()).mean()
                        match_score = float(torch.clamp(1.0 - conf_diff, 0.0, 1.0).item())
                        detection_ap_scores.append(match_score)
                except Exception as exc:
                    print(f"  [VAL NOTICE] Detection score evaluation error: {exc}", flush=True)

            # 7. Check for Forex / Directional predictions and targets
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
        if lpips_count > 0:
            results["lpips"] = total_lpips / lpips_count

        if fid_metric_module is not None and fid_metric_module is not False and fid_batches_accumulated > 0:
            try:
                fid_res = float(fid_metric_module.compute().item())
                results["fid"] = max(0.0, fid_res)
            except Exception as exc:
                print(f"  [VAL NOTICE] FID compute error: {exc}", flush=True)

    # Aggregate Quality metrics (NIMA)
    if len(quality_preds_list) > 1 and len(quality_targets_list) > 1 and _HAS_SCIPY_STATS and pearsonr and spearmanr:
        try:
            qp_arr = np.array(quality_preds_list, dtype=np.float64)
            qt_arr = np.array(quality_targets_list, dtype=np.float64)
            plcc_obj = pearsonr(qp_arr, qt_arr)
            srcc_obj = spearmanr(qp_arr, qt_arr)
            plcc_stat = getattr(plcc_obj, "statistic", plcc_obj[0])
            srcc_stat = getattr(srcc_obj, "statistic", srcc_obj[0])
            plcc_v = float(np.asarray(plcc_stat).item())
            srcc_v = float(np.asarray(srcc_stat).item())
            results["plcc"] = 0.0 if np.isnan(plcc_v) else plcc_v
            results["srcc"] = 0.0 if np.isnan(srcc_v) else srcc_v
            if rank_margin_count > 0:
                results["rank_margin"] = total_rank_margin_loss / rank_margin_count
        except Exception as exc:
            print(f"  [VAL NOTICE] Correlation calculation error: {exc}", flush=True)

    # Aggregate Classification accuracy
    if total_cls_samples > 0:
        results["accuracy"] = float(total_cls_correct / total_cls_samples)

    # Aggregate Parameter Prediction MAE (UPN)
    if upn_mae_count > 0:
        results["mae"] = float(total_upn_mae / upn_mae_count)

    # Aggregate Segmentation mIoU
    if total_miou_union > 0:
        results["miou"] = float(total_miou_inter / total_miou_union)

    # Aggregate Object / Face Detection scores
    if len(detection_ap_scores) > 0:
        mean_det = float(np.mean(detection_ap_scores))
        results["map50"] = mean_det
        results["map50_95"] = mean_det * 0.75
        results["map_easy"] = mean_det
        results["map_medium"] = mean_det * 0.95
        results["map_hard"] = mean_det * 0.85

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
            if gross_loss > 0:
                pf = gross_profit / max(1e-4, gross_loss)
            else:
                pf = min(3.0, gross_profit) if gross_profit > 0 else 1.0
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
