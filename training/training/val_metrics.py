"""Per-batch validation metric updaters, one per metric family.

Each updater reads one batch of predictions and targets and adds to the running totals in
:class:`ValAccumulators`. :data:`BATCH_METRICS` lists them in evaluation order; the
classification and parameter-prediction updaters read the Forex flag set by earlier batches,
so the Forex updater runs last.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from data.sample_labels import FACE_TARGET_LEN
from training.ignored_targets import labelled
from training.training.amp import compute_ssim_gpu
from training.training.face_detection_metrics import (
    FACE_DETECTION_METRICS,
    FacePrediction,
    record_face_batch,
)
from training.training.forex_equity import risk_multiple
from training.training.val_accumulators import ImageMetricModules, ValAccumulators
from training.utils.error_types import METRIC_INIT_ERRORS
from training.utils.lazy_imports import load_fid_metric, load_lpips_metric, load_optional

logger = logging.getLogger("lemtrain.training.validation")

# Failure modes of per-batch metric math on model outputs (shape/dtype mismatch, CUDA/OOM, bad
# indexing).
METRIC_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, ValueError, TypeError, IndexError)

LearnedPerceptualImagePatchSimilarity = load_optional(load_lpips_metric)
FrechetInceptionDistance = load_optional(load_fid_metric)


@dataclass
class ValBatch:
    """One evaluated batch and the evaluation settings that apply to it."""

    preds: Any
    targets: Any
    sota_cfg: dict[str, Any]
    dataset_type: Any
    device: torch.device
    governor: Any
    # Causal-LM token labels (``-100`` = ignored); None for non-LLM models.
    labels: Any = None


def _relieve_metric_oom(governor: Any) -> None:
    if hasattr(governor, "handle_oom"):
        governor.handle_oom()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lpips_module(modules: ImageMetricModules, device: torch.device) -> Any:
    """Build the LPIPS module on first use; returns None or False when unavailable."""
    if modules.lpips is None:
        try:
            if LearnedPerceptualImagePatchSimilarity is None:
                raise ImportError("TorchMetrics LPIPS is not installed")
            modules.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",
                normalize=False,
            ).to(device)
            modules.lpips.eval()
            for p_param in modules.lpips.parameters():
                p_param.requires_grad = False
        except METRIC_INIT_ERRORS as exc:
            logger.warning("LPIPS metric module initialization failed: %s", exc)
            modules.lpips = False
    return modules.lpips


def _fid_module(modules: ImageMetricModules, device: torch.device) -> Any:
    """Build the 2048-feature FID module on first use; returns None or False when unavailable."""
    if modules.fid is None:
        try:
            if FrechetInceptionDistance is None:
                raise ImportError("TorchMetrics FID is not installed")
            modules.fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                sync_on_compute=False,
            ).to(device)
            modules.fid.eval()
        except METRIC_INIT_ERRORS as exc:
            logger.warning("FID metric module initialization failed: %s", exc)
            modules.fid = False
    return modules.fid


def _update_lpips(
    acc: ValAccumulators, batch: ValBatch, pred: torch.Tensor, target: torch.Tensor
) -> None:
    """LPIPS in FP32, one image per forward pass."""
    metric = _lpips_module(acc.modules, batch.device)
    if metric is None or metric is False:
        return
    try:
        p_scaled = (torch.clamp(pred, 0.0, 1.0) * 2.0 - 1.0).float()
        t_scaled = (torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0).float()
        b_sz = p_scaled.size(0)
        chunk_lpips_sum = 0.0
        for m_idx in range(b_sz):
            p_chunk = p_scaled[m_idx : m_idx + 1]
            t_chunk = t_scaled[m_idx : m_idx + 1]
            chunk_lpips_sum += float(metric(p_chunk, t_chunk).item()) * p_chunk.size(0)
        acc.image.total_lpips += chunk_lpips_sum / max(1, b_sz)
        acc.image.lpips_count += 1
    except METRIC_ERRORS as exc:
        logger.warning("LPIPS evaluation error: %s", exc)
        _relieve_metric_oom(batch.governor)


FID_CHUNK_SIZE = 2
"""Images per FID update call (publication-grade FID invariant: at most 2)."""


def _update_fid(
    acc: ValAccumulators, batch: ValBatch, pred: torch.Tensor, target: torch.Tensor
) -> None:
    """Accumulate FID features in micro-chunks of at most 2 images."""
    metric = _fid_module(acc.modules, batch.device)
    if metric is None or metric is False:
        return
    try:
        p_u8 = torch.clamp(pred.detach() * 255.0, 0.0, 255.0).to(torch.uint8)
        t_u8 = torch.clamp(target.detach() * 255.0, 0.0, 255.0).to(torch.uint8)
        if p_u8.shape[1] == 1:
            p_u8 = p_u8.repeat(1, 3, 1, 1)
            t_u8 = t_u8.repeat(1, 3, 1, 1)
        _update_fid_pairs(acc, metric, t_u8, p_u8)
        acc.image.fid_batches_accumulated += 1
    except METRIC_ERRORS as exc:
        logger.warning("FID update error: %s", exc)


def _update_fid_pairs(
    acc: ValAccumulators, metric: Any, real_u8: torch.Tensor, fake_u8: torch.Tensor
) -> None:
    """Feed real and fake chunks pairwise; a pair cut in half marks the pass's FID incomplete."""
    for c_idx in range(0, fake_u8.size(0), FID_CHUNK_SIZE):
        metric.update(real_u8[c_idx : c_idx + FID_CHUNK_SIZE], real=True)
        try:
            metric.update(fake_u8[c_idx : c_idx + FID_CHUNK_SIZE], real=False)
        except METRIC_ERRORS:
            acc.image.fid_incomplete = True
            logger.warning("FID fake-feature update failed after its real half; FID is dropped.")
            raise


def _image_pair(batch: ValBatch) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return same-shape 4D prediction and target images, or None."""
    preds, targets = batch.preds, batch.targets
    if isinstance(preds, (tuple, list)) and len(preds) > 0:
        preds = preds[0]
    pred = preds.get("restored_image", preds) if isinstance(preds, dict) else preds
    if isinstance(targets, (tuple, list)) and len(targets) > 0:
        targets = targets[0]
    target = targets.get("image", targets) if isinstance(targets, dict) else targets
    if not (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)):
        return None
    if pred.ndim == 4 and target.ndim == 4 and pred.shape == target.shape:
        return pred, target
    return None


def update_image(acc: ValAccumulators, batch: ValBatch) -> None:
    """Vision restoration / enhancement: MSE (for PSNR), SSIM, LPIPS and FID."""
    pair = _image_pair(batch)
    if pair is None:
        return
    pred, target = pair
    acc.image.has_image_metrics = True
    acc.image.total_mse += float(torch.nn.functional.mse_loss(pred, target).item())
    try:
        acc.image.total_ssim += float(compute_ssim_gpu(pred, target).item())
    except METRIC_ERRORS as exc:
        logger.warning("SSIM evaluation error; batch excluded from SSIM: %s", exc)
    if "lpips" in batch.sota_cfg and LearnedPerceptualImagePatchSimilarity is not None:
        _update_lpips(acc, batch, pred, target)
    if "fid" in batch.sota_cfg and FrechetInceptionDistance is not None:
        _update_fid(acc, batch, pred, target)


def _is_score_distribution(batch: ValBatch) -> bool:
    """True for quality models whose predictions and targets are 10-bin score histograms."""
    is_quality_model = (
        any(k in batch.sota_cfg for k in ("plcc", "srcc", "rank_margin"))
        or batch.dataset_type == "quality"
    )
    preds, targets = batch.preds, batch.targets
    if not (
        is_quality_model and isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor)
    ):
        return False
    return preds.ndim >= 2 and targets.ndim >= 2 and min(preds.shape[-1], targets.shape[-1]) >= 10


def _update_rank_margin(
    acc: ValAccumulators, scores: tuple[torch.Tensor, torch.Tensor], margin_val: float
) -> None:
    """Pairwise hinge loss of predicted mean scores against the target ordering."""
    p_m_t, t_m_t = scores
    p_diff = p_m_t.unsqueeze(0) - p_m_t.unsqueeze(1)
    t_sign = torch.sign(t_m_t.unsqueeze(0) - t_m_t.unsqueeze(1))
    rank_loss = torch.relu(margin_val - t_sign * p_diff)
    mask = (t_sign != 0).float()
    if mask.sum() > 0:
        acc.quality.total_rank_margin_loss += float(
            (rank_loss * mask).sum().item() / mask.sum().item()
        )
        acc.quality.rank_margin_count += 1


def update_quality(acc: ValAccumulators, batch: ValBatch) -> None:
    """Quality assessment and aesthetic ranking (NIMA): mean scores and rank-margin loss."""
    if not _is_score_distribution(batch):
        return
    preds, targets = batch.preds, batch.targets
    try:
        p_probs = torch.softmax(preds.float(), dim=-1)
        t_probs = targets.float() / torch.clamp(targets.float().sum(dim=-1, keepdim=True), min=1e-6)
        rank_w = torch.arange(1, 11, dtype=torch.float32, device=preds.device)
        p_mean = (p_probs * rank_w).sum(dim=-1).detach().cpu().numpy().tolist()
        t_mean = (t_probs * rank_w).sum(dim=-1).detach().cpu().numpy().tolist()
        acc.quality.quality_preds_list.extend(p_mean)
        acc.quality.quality_targets_list.extend(t_mean)
        if preds.size(0) > 1 and "rank_margin" in batch.sota_cfg:
            _update_rank_margin(
                acc,
                ((p_probs * rank_w).sum(dim=-1), (t_probs * rank_w).sum(dim=-1)),
                float(batch.sota_cfg.get("rank_margin", 0.05)),
            )
    except METRIC_ERRORS as exc:
        logger.warning("Quality metrics evaluation error: %s", exc)


def _tensor_pair(batch: ValBatch) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(batch.preds, torch.Tensor) and isinstance(batch.targets, torch.Tensor):
        return batch.preds, batch.targets
    return None


def update_classification(acc: ValAccumulators, batch: ValBatch) -> None:
    """Categorical classification accuracy (NSFW, authenticity); skipped for Forex models."""
    if "accuracy" not in batch.sota_cfg or acc.direction.has_forex_metrics:
        return
    try:
        pair = _tensor_pair(batch)
        if pair is None:
            return
        preds, targets = pair
        if preds.ndim >= 2 and (targets.ndim == 1 or targets.shape[-1] == 1):
            pred_labels = torch.argmax(preds, dim=-1).reshape(-1)
            tgt_labels = targets.reshape(-1).long()
            # Unlabeled samples (IGNORE_TARGET) count neither as right nor as wrong.
            known = labelled(tgt_labels)
            hits = (pred_labels == tgt_labels) & known
            acc.tasks.total_cls_correct += int(hits.sum().item())
            acc.tasks.total_cls_samples += int(known.sum().item())
        elif preds.ndim == 1 and targets.ndim == 1:
            pred_bin = (torch.sigmoid(preds) > 0.5).long()
            acc.tasks.total_cls_correct += int((pred_bin == targets.long()).sum().item())
            acc.tasks.total_cls_samples += targets.numel()
    except METRIC_ERRORS as exc:
        logger.warning("Classification accuracy evaluation error: %s", exc)


def update_parameter_mae(acc: ValAccumulators, batch: ValBatch) -> None:
    """Parameter prediction (UPN) mean absolute error; skipped for Forex models."""
    if "mae" not in batch.sota_cfg or acc.direction.has_forex_metrics:
        return
    try:
        pair = _tensor_pair(batch)
        if pair is not None:
            preds, targets = pair
            upn_err = torch.abs(preds.float() - targets.float()).mean().item()
            acc.tasks.total_upn_mae += float(upn_err)
            acc.tasks.upn_mae_count += 1
    except METRIC_ERRORS as exc:
        logger.warning("UPN MAE evaluation error: %s", exc)


def update_segmentation(acc: ValAccumulators, batch: ValBatch) -> None:
    """Semantic segmentation (ParseNet) intersection and union counts for mIoU."""
    if "miou" not in batch.sota_cfg:
        return
    try:
        pair = _tensor_pair(batch)
        if pair is not None:
            preds, targets = pair
            pred_seg = torch.argmax(preds, dim=1) if preds.ndim == 4 else preds
            tgt_seg = targets.squeeze(1) if targets.ndim == 4 else targets
            inter = torch.logical_and(tgt_seg > 0, pred_seg == tgt_seg).sum().item()
            union = torch.logical_or(tgt_seg > 0, pred_seg > 0).sum().item()
            acc.tasks.total_miou_inter += float(inter)
            acc.tasks.total_miou_union += float(union)
    except METRIC_ERRORS as exc:
        logger.warning("Segmentation mIoU evaluation error: %s", exc)


def _face_prediction(batch: ValBatch) -> FacePrediction | None:
    """The RetinaFace ``(boxes, confidence, landmarks)`` output of a batch with face targets."""
    preds, targets = batch.preds, batch.targets
    if not (
        isinstance(preds, (tuple, list))
        and len(preds) >= 3
        and all(isinstance(p, torch.Tensor) for p in preds[:3])
        and isinstance(targets, torch.Tensor)
        and targets.ndim == 2
        and targets.shape[-1] == FACE_TARGET_LEN
    ):
        return None
    return FacePrediction(boxes=preds[0], confidence=preds[1], landmarks=preds[2])


def update_face_detection(acc: ValAccumulators, batch: ValBatch) -> None:
    """Single-box face detection (RetinaFace): box IoU, confidence ranking and landmark NME."""
    if not any(k in batch.sota_cfg for k in FACE_DETECTION_METRICS):
        return
    prediction = _face_prediction(batch)
    if prediction is None:
        return
    try:
        record_face_batch(acc.faces, prediction, batch.targets)
    except METRIC_ERRORS as exc:
        logger.warning("Face detection metric evaluation error: %s", exc)


def _sequence_pair(value: Any) -> tuple[Any, Any] | None:
    """The first two entries of a sequence (non-tensors become None), or None."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        first = value[0] if isinstance(value[0], torch.Tensor) else None
        second = value[1] if isinstance(value[1], torch.Tensor) else None
        return first, second
    return None


def extract_forex_tensors(
    preds: Any,
    targets: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Extract directional logits, magnitude predictions, and targets."""
    if isinstance(preds, dict):
        pred_pair = (preds.get("direction_logits", preds.get("direction")), preds.get("magnitude"))
    else:
        pred_pair = _sequence_pair(preds) or (None, None)
    if isinstance(targets, dict):
        target_pair = (targets.get("direction"), targets.get("magnitude"))
    else:
        target_pair = _sequence_pair(targets)
    if target_pair is None:
        label_vector = targets if isinstance(targets, torch.Tensor) and targets.ndim == 1 else None
        target_pair = (label_vector, None)
    return pred_pair[0], pred_pair[1], target_pair[0], target_pair[1]


def _update_magnitude(
    acc: ValAccumulators, mag_preds: torch.Tensor, mag_targets: torch.Tensor, bs: int
) -> None:
    clamped_mag = torch.clamp(mag_preds, max=100.0)
    if min(clamped_mag.ndim, mag_targets.ndim) >= 2 and (
        min(clamped_mag.shape[-1], mag_targets.shape[-1]) >= 2
    ):
        tp_err = torch.abs(clamped_mag[..., 0] - mag_targets[..., 0]).sum().item()
        sl_err = torch.abs(clamped_mag[..., 1] - mag_targets[..., 1]).sum().item()
        acc.trades.total_tp_mae += float(tp_err)
        acc.trades.total_sl_mae += float(sl_err)
        acc.trades.total_mag_samples += bs


def _simulate_trades(
    acc: ValAccumulators,
    pred_dir: torch.Tensor,
    dir_targets: torch.Tensor,
    mag_targets: torch.Tensor | None,
) -> None:
    """Append the simulated PnL of every non-HOLD prediction (win: +TP, loss: -SL)."""
    directional_mask = pred_dir != 1
    if not directional_mask.any():
        return
    dirs_np = pred_dir[directional_mask].detach().cpu().numpy()
    targets_np = dir_targets[directional_mask].detach().cpu().numpy()
    mags_np = (
        mag_targets[directional_mask].detach().cpu().numpy()
        if mag_targets is not None and mag_targets.ndim >= 2
        else None
    )
    for idx, pred_d in enumerate(dirs_np):
        tp_val = float(mags_np[idx, 0]) if mags_np is not None else 15.0
        sl_val = float(mags_np[idx, 1]) if mags_np is not None else 15.0
        won = int(pred_d) == int(targets_np[idx])
        acc.trades.trade_returns.append(tp_val if won else -sl_val)
        multiple = risk_multiple(tp_val, sl_val, won)
        if multiple is not None:
            acc.trades.trade_risk_multiples.append(multiple)


def update_forex(acc: ValAccumulators, batch: ValBatch) -> None:
    """Forex direction accuracy, trades, entropy, TP/SL error and simulated returns."""
    dir_logits, mag_preds, dir_targets, mag_targets = extract_forex_tensors(
        batch.preds, batch.targets
    )
    if dir_logits is None or dir_targets is None:
        return
    totals = acc.direction
    totals.has_forex_metrics = True
    pred_dir = torch.argmax(dir_logits, dim=-1)
    bs = dir_targets.size(0)
    totals.total_forex_samples += bs
    totals.correct_forex_samples += int((pred_dir == dir_targets).sum().item())

    # Directional trades: non-HOLD predictions (0 = SELL, 2 = BUY)
    directional_mask = pred_dir != 1
    totals.total_trades += int(directional_mask.sum().item())
    totals.winning_trades += int(((pred_dir == dir_targets) & directional_mask).sum().item())

    probs = torch.softmax(dir_logits, dim=-1)
    batch_entropy = float(-(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item())
    totals.total_forex_entropy += batch_entropy * bs

    if mag_preds is not None and mag_targets is not None:
        _update_magnitude(acc, mag_preds, mag_targets, bs)
    _simulate_trades(acc, pred_dir, dir_targets, mag_targets)


def update_llm_tokens(acc: ValAccumulators, batch: ValBatch) -> None:
    """Next-token accuracy of causal LLMs over the non-ignored (``!= -100``) labels."""
    preds, labels = batch.preds, batch.labels
    if not (
        isinstance(preds, torch.Tensor)
        and isinstance(labels, torch.Tensor)
        and preds.dim() == 3
        and labels.shape == preds.shape[:2]
    ):
        return
    shift_preds = preds[..., :-1, :].argmax(dim=-1)
    shift_targets = labels[..., 1:]
    mask = shift_targets != -100
    if not bool(mask.any()):
        return
    correct = (shift_preds[mask] == shift_targets[mask]).sum()
    acc.llm.total_llm_correct_tokens += int(correct.item())
    acc.llm.total_llm_valid_tokens += int(mask.sum().item())


BATCH_METRICS: tuple[Callable[[ValAccumulators, ValBatch], None], ...] = (
    update_image,
    update_quality,
    update_classification,
    update_parameter_mae,
    update_segmentation,
    update_face_detection,
    update_llm_tokens,
    update_forex,
)
