"""Training loss functions: forex dual loss, focal loss, and the unified CombinedLoss engine."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from data.dataset_targets import multitask_head
from data.sample_labels import FACE_BOX_SLICE, FACE_LANDMARK_SLICE, LANDMARK_ABSENT
from training.utils.lazy_imports import load_lpips_metric, load_optional

LearnedPerceptualImagePatchSimilarity = load_optional(load_lpips_metric)

logger = logging.getLogger("lemtrain.losses")

_AUTOCAST_DEVICE_TYPES = ("cuda", "mps", "cpu", "xpu")

# ─────────────────────────────────────────────────────────────────────────────
# Forex Dual Loss
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Multitask Restoration Head Registry
# ─────────────────────────────────────────────────────────────────────────────

MULTITASK_HEAD_NAMES: tuple[str, ...] = (
    "denoise",  # 0
    "deblur",  # 1
    "derain",  # 2
    "dehaze_indoor",  # 3
    "dehaze_outdoor",  # 4
    "lowlight",  # 5
    "exposure",  # 6
    "superres",  # 7
    "vintage",  # 8
    "face_restorer",  # 9
    "face_parser",  # 10
)

MULTITASK_HEAD_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(MULTITASK_HEAD_NAMES)
}


def _resolve_task_index_tensor(
    task_idx: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve varied task representations into an integer class index tensor.

    Handles torch.Tensor (int or float), sequence of strings/integers (from DataLoader collation),
    single string, or None. Returns None if task_idx cannot be resolved.
    """
    if task_idx is None:
        return None

    if isinstance(task_idx, torch.Tensor):
        return task_idx.to(device=device, dtype=torch.long) if task_idx.numel() > 0 else None

    if isinstance(task_idx, (list, tuple)):
        indices: list[int] = []
        for item in task_idx:
            if isinstance(item, int):
                indices.append(item)
            elif isinstance(item, str):
                canonical = multitask_head(item)
                indices.append(MULTITASK_HEAD_TO_INDEX.get(canonical, 0))
            elif isinstance(item, torch.Tensor):
                indices.append(int(item.item()))
            else:
                indices.append(0)
        return torch.tensor(indices, dtype=torch.long, device=device)

    if isinstance(task_idx, str):
        canonical = multitask_head(task_idx)
        idx = MULTITASK_HEAD_TO_INDEX.get(canonical, 0)
        return torch.full((batch_size,), idx, dtype=torch.long, device=device)

    if isinstance(task_idx, (int, float)):
        return torch.full((batch_size,), int(task_idx), dtype=torch.long, device=device)

    return None


@dataclass(frozen=True)
class ForexLossWeights:
    """Component weights and shape parameters of :class:`ForexDualLoss` (see its Args)."""

    direction_weight: float = 0.5
    magnitude_weight: float = 0.02
    huber_delta: float = 2.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05


class ForexDualLoss(nn.Module):
    """
    LemGendary Forex Dual Loss Engine v2.0.

    Combines:
        - Focal Loss for direction (Down / Sideways / Up)
          Focal loss (gamma=2.0) down-weights easy Sideways predictions and forces
          attention to hard Buy/Sell boundary cases, preventing class collapse.
        - Huber (SmoothL1) for magnitude (TP pips, SL pips)
          Magnitude predictions are clamped to 200 pips before loss computation.

    Direction confidence gates magnitude loss:
        Low-confidence bars (high entropy in direction logits) contribute
        proportionally less to the magnitude regression signal, preventing
        the magnitude head from fitting noise on ambiguous bars.

    Governor integration:
        softmax_temp and logit_clamp from the Governor stabilizer dict are applied
        to direction logits so the Governor's temperature/clamp adjustments are
        reflected in the forex training signal.

    Args (fields of ``weights``):
        direction_weight: Weight for Focal loss component (default 1.0).
        magnitude_weight: Weight for Huber loss component (default 0.5).
        huber_delta:      Huber delta -- transitions L2->L1 at this pip threshold.
        focal_gamma:      Focal loss focusing parameter (default 2.0).
        label_smoothing:  Label smoothing for direction targets (default 0.05).
    """

    class_weights: torch.Tensor  # registered buffer

    def __init__(
        self,
        weights: ForexLossWeights | None = None,
        *,
        stabilizers: dict | None = None,
    ):
        super().__init__()
        weights = weights or ForexLossWeights()
        self.direction_weight = weights.direction_weight
        self.magnitude_weight = weights.magnitude_weight
        self.huber_delta = weights.huber_delta
        self.focal_gamma = weights.focal_gamma
        self.label_smoothing = weights.label_smoothing
        self.max_pips = 100.0
        self.stab = stabilizers or {}
        # Asymmetric class weights: stronger Sideways penalty to prevent collapse
        # Weight vector: [Down=1.3, Sideways=0.7, Up=1.3]
        self.register_buffer("class_weights", torch.tensor([1.3, 0.7, 1.3]))

    def forward(
        self, pred: dict, labels: dict, _task_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            pred:   Dict with 'direction_logits' [B, 3] and 'magnitude' [B, 2]
            labels: Dict with 'direction' [B] (long) and 'magnitude' [B, 2] (float)
            _task_idx: Accepted for the shared criterion(preds, targets, task_idx)
                call protocol used by the epoch/validation loops; unused here.

        Returns:
            Combined scalar loss.
        """
        dir_logits = (
            pred["direction_logits"] if "direction_logits" in pred else pred["direction"]
        )  # [B, 3]
        dir_logits = self._stabilize(dir_logits)
        dir_loss = self._focal_direction_loss(dir_logits, labels["direction"])  # [B] long target

        # Confidence gate: high-entropy bars get lower magnitude weight
        conf_gate = _confidence_gate(
            dir_logits, float(getattr(self, "conf_gate_strength", 1.0))
        )  # [B]

        # Clamp magnitude predictions before Huber to prevent runaway pip gradients
        mag_pred_clamped = torch.clamp(pred["magnitude"], max=self.max_pips)  # [B, 2]

        # Huber loss for magnitude, gated by direction confidence
        huber = F.smooth_l1_loss(
            mag_pred_clamped, labels["magnitude"], reduction="none", beta=self.huber_delta
        )  # [B, 2]
        mag_loss = (huber.mean(dim=1) * conf_gate).mean()

        return self.direction_weight * dir_loss + self.magnitude_weight * mag_loss

    def _stabilize(self, dir_logits: torch.Tensor) -> torch.Tensor:
        """Governor stabilizers: logit clamping, then temperature scaling."""
        stab = getattr(self, "stab", {})
        softmax_temp = float(stab.get("softmax_temp", 1.0)) if stab else 1.0
        logit_clamp = float(stab.get("logit_clamp", 15.0)) if stab else 15.0
        if logit_clamp > 0:
            dir_logits = torch.clamp(dir_logits, min=-logit_clamp, max=logit_clamp)
        if softmax_temp != 1.0 and softmax_temp > 0:
            dir_logits = dir_logits / softmax_temp
        return dir_logits

    def _focal_direction_loss(
        self, dir_logits: torch.Tensor, dir_target: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for direction: down-weights easy Sideways, focuses on hard Buy/Sell."""
        ce_loss = F.cross_entropy(
            dir_logits,
            dir_target,
            weight=self.class_weights.to(dir_logits.device),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.focal_gamma) * ce_loss).mean()


def _confidence_gate(dir_logits: torch.Tensor, strength: float) -> torch.Tensor:
    """``(1 - normalised entropy) ** strength`` per bar, without gradient."""
    with torch.no_grad():
        probs = torch.softmax(dir_logits, dim=-1)  # [B, 3]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B]
        max_ent = torch.log(torch.tensor(3.0, device=probs.device))
        return (1.0 - (entropy / max_ent).clamp(0.0, 1.0)) ** strength  # [B]


class SoftSpearmanLoss(nn.Module):
    """
    Differentiable Soft-Spearman Rank Correlation Loss.
    Approximates the rank operator using temperature-scaled sigmoid pairwise comparisons:
        r_i = 1 + \\sum_{j \\ne i} \\sigma((p_i - p_j) / \\tau)
    Optimizes Spearman rank correlation directly with smooth gradients.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, p_scores: torch.Tensor, t_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_scores: Predicted 1D continuous scores [N]
            t_scores: Target 1D continuous scores [N]
        Returns:
            Scalar loss: 1 - soft_spearman_correlation
        """
        n = p_scores.size(0)
        if n < 2:
            # N=1 guard: rank correlation is undefined for a single sample
            return torch.tensor(0.0, device=p_scores.device, requires_grad=True)

        # Pairwise difference matrices: [N, N]
        p_diff = (p_scores.unsqueeze(0) - p_scores.unsqueeze(1)) / self.temperature
        t_diff = (t_scores.unsqueeze(0) - t_scores.unsqueeze(1)) / self.temperature

        # Soft ranks via sigmoid
        p_ranks = 1.0 + torch.sigmoid(p_diff).sum(dim=1)
        t_ranks = 1.0 + torch.sigmoid(t_diff).sum(dim=1)

        # Center ranks
        p_ranks_c = p_ranks - p_ranks.mean()
        t_ranks_c = t_ranks - t_ranks.mean()

        # Pearson correlation of soft ranks
        cov = (p_ranks_c * t_ranks_c).sum()
        var_p = (p_ranks_c**2).sum()
        var_t = (t_ranks_c**2).sum()

        # N=1 variance guard: both variances must be non-zero
        if var_p < self.eps or var_t < self.eps:
            return torch.tensor(0.0, device=p_scores.device, requires_grad=True)

        denom = torch.sqrt(torch.clamp(var_p * var_t, min=self.eps))
        soft_spearman = cov / denom

        return 1.0 - soft_spearman


class RankMemoryBank:
    """
    Cross-Microbatch FIFO Memory Bank for Rank & Contrastive Supervision.
    Caches detached predictions and targets across gradient accumulation steps,
    allowing effective pairwise ranking over (N_batch + N_memory) samples
    even when local VRAM forces micro-batch size b=2.
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.p_memory = []
        self.t_memory = []

    def get_context(self, p_active: torch.Tensor, t_active: torch.Tensor):
        """
        Concatenates active batch with detached historical representations.
        Gradients backpropagate exclusively through p_active.
        """
        if len(self.p_memory) == 0:
            return p_active, t_active

        p_hist = torch.stack(self.p_memory).to(p_active.device)
        t_hist = torch.stack(self.t_memory).to(t_active.device)

        p_full = torch.cat([p_active, p_hist], dim=0)
        t_full = torch.cat([t_active, t_hist], dim=0)
        return p_full, t_full

    def update(self, p_active: torch.Tensor, t_active: torch.Tensor):
        """Push active samples into detached FIFO queue."""
        with torch.no_grad():
            for p, t in zip(p_active.detach().view(-1), t_active.detach().view(-1), strict=False):
                self.p_memory.append(p)
                self.t_memory.append(t)
                if len(self.p_memory) > self.capacity:
                    self.p_memory.pop(0)
                    self.t_memory.pop(0)

    def reset(self):
        """Clear memory at epoch boundary or resolution jump."""
        self.p_memory.clear()
        self.t_memory.clear()


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for Safety / NSFW Categorical Classification.
    Down-weights easy well-classified negatives to focus on hard boundary triggers.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean focal-weighted cross-entropy of ``pred`` logits vs ``target``."""
        ce_loss = F.cross_entropy(
            pred, target, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def _correlation_proxy(p_eval: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
    """``1 - Pearson correlation`` of predicted and target mean scores."""
    p_c = p_eval - p_eval.mean()
    t_c = t_eval - t_eval.mean()
    denom_ssim = torch.sqrt(torch.clamp((p_c**2).sum() * (t_c**2).sum(), min=1e-8))
    return 1.0 - (p_c * t_c).sum() / denom_ssim


def _pairwise_rank_loss(p_eval: torch.Tensor, t_eval: torch.Tensor, margin: float) -> torch.Tensor:
    """Mean margin hinge over all pairs whose target order is strict."""
    p_diff = p_eval.unsqueeze(0) - p_eval.unsqueeze(1)
    t_sign = torch.sign(t_eval.unsqueeze(0) - t_eval.unsqueeze(1))
    rank_loss = F.relu(margin - t_sign * p_diff)
    mask = (t_sign != 0).float()
    return (rank_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def autocast_device_type(device: torch.device) -> str:
    """``torch.autocast`` device type of ``device`` (``cpu`` for unsupported backends)."""
    return device.type if device.type in _AUTOCAST_DEVICE_TYPES else "cpu"


def _face_detection_loss(pred: Any, target: torch.Tensor) -> torch.Tensor:
    """Confidence BCE plus bbox (x2) and landmark Huber losses on positive anchors.

    Landmark coordinates marked ``LANDMARK_ABSENT`` (box-only labels) are left out of the
    landmark loss instead of pulling predictions towards made-up points.
    """
    p_bbox, p_conf, p_landm = pred
    t_conf = target[:, 0:1]

    with torch.autocast(device_type=autocast_device_type(p_conf.device), enabled=False):
        loss_conf = F.binary_cross_entropy(p_conf.float(), t_conf.float())

    pos = (t_conf > 0.5).squeeze(-1)
    loss_bbox = torch.tensor(0.0, device=p_bbox.device)
    loss_landm = torch.tensor(0.0, device=p_landm.device)
    if pos.any():
        loss_bbox = F.smooth_l1_loss(p_bbox[pos], target[:, FACE_BOX_SLICE][pos])
        t_landm = target[:, FACE_LANDMARK_SLICE][pos]
        present = t_landm > LANDMARK_ABSENT / 2.0
        if present.any():
            loss_landm = F.smooth_l1_loss(p_landm[pos][present], t_landm[present])

    return loss_conf + loss_bbox * 2.0 + loss_landm * 1.0


def _is_forex_prediction(pred) -> bool:
    """True for a Forex model output: a dict carrying direction logits."""
    return isinstance(pred, dict) and ("direction" in pred or "direction_logits" in pred)


def _restoration_output(pred) -> dict | None:
    """``pred`` when it is a dict model output carrying ``restored_image``, else None."""
    return pred if isinstance(pred, dict) and "restored_image" in pred else None


@dataclass(frozen=True)
class _BaseCriteria:
    """Stateless base criteria of :class:`CombinedLoss`.

    None of them has parameters or buffers, so keeping them outside the module tree leaves
    ``CombinedLoss.state_dict()`` keys unchanged.
    """

    l1: nn.L1Loss = field(default_factory=lambda: nn.L1Loss(reduction="mean"))
    # Legacy fallback for face and segmentation topology
    mse: nn.MSELoss = field(default_factory=lambda: nn.MSELoss(reduction="mean"))
    ce: nn.CrossEntropyLoss = field(default_factory=lambda: nn.CrossEntropyLoss(ignore_index=255))
    focal: FocalLoss = field(default_factory=lambda: FocalLoss(gamma=2.0, label_smoothing=0.05))
    soft_spearman: SoftSpearmanLoss = field(
        default_factory=lambda: SoftSpearmanLoss(temperature=0.1)
    )


class CombinedLoss(nn.Module):
    """
    LemGendary 2026 Unified Loss Engine
    Supports Restoration (L1+LPIPS), Quality (EMD+SoftSpearman+RankBoost),
    and Classification (CE/Focal).
    """

    # Registered buffers.
    rank_weights: torch.Tensor
    task_lpips_weights: torch.Tensor

    def __init__(self, task_type="restoration", stabilizers=None, use_perc=False):
        super().__init__()
        self.task_type = task_type
        # 2026 Resilience: Dynamic injection from config hierarchy
        self.stab = stabilizers or {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0}
        self.criteria = _BaseCriteria()
        self.forex_dual = ForexDualLoss(stabilizers=self.stab)
        self.memory_bank = None
        if self.stab.get("rank_memory_bank_size", 0) > 0 or self.stab.get("use_memory_bank", False):
            mb_size = int(self.stab.get("rank_memory_bank_size", 32))
            self.memory_bank = RankMemoryBank(capacity=mb_size)
        self.perc = None

        # 2026: SOTA Rank-Boost Weights (Standard 10..1 mapping)
        self.register_buffer("rank_weights", torch.arange(1, 11).float())

        # 2026: MoE 11-Manifold Perceptual Scaling Coefficients
        # Balances LPIPS gradients to prevent high-frequency tasks (Face, SuperRes) from
        # overwhelming intensity-based tasks (Denoise, Lowlight).
        self.register_buffer(
            "task_lpips_weights",
            torch.tensor(
                [
                    0.010,  # 0: denoise
                    0.025,  # 1: deblur
                    0.020,  # 2: derain
                    0.015,  # 3: dehaze_indoor
                    0.015,  # 4: dehaze_outdoor
                    0.010,  # 5: lowlight
                    0.010,  # 6: exposure
                    0.030,  # 7: superres
                    0.050,  # 8: vintage
                    0.050,  # 9: face_restorer
                    0.005,  # 10: face_parser
                ],
                dtype=torch.float32,
            ),
        )

        if self.task_type in ["restoration", "enhancement"] and use_perc:
            try:
                # Reclaim memory before loading the heavy perceptual engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    " [MISSION] Initializing Neural Perceptual Engine (TorchMetrics LPIPS/VGG)..."
                )

                class AutocastLPIPS(torch.nn.Module):
                    """Wrap an LPIPS module so its forward pass runs under CUDA autocast."""

                    def __init__(self, perc_module):
                        super().__init__()
                        self.perc = perc_module

                    def forward(self, x, y):
                        """Return the LPIPS distance between ``x`` and ``y`` under autocast."""
                        with torch.autocast(
                            device_type=autocast_device_type(x.device), enabled=x.is_cuda
                        ):
                            return self.perc(x, y)

                if LearnedPerceptualImagePatchSimilarity is None:
                    raise ImportError("TorchMetrics LPIPS is not installed")

                self.perc = AutocastLPIPS(
                    LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False)
                )

                self.perc.eval()
                for param in self.perc.parameters():
                    param.requires_grad = False
            except ImportError:
                print(
                    "[WARNING] [RESILIENCE] TorchMetrics not found. "
                    "Run: pip install torchmetrics[image]"
                )
                print("[WARNING] [RESILIENCE] Defaulting to pure L1.")
            except (OSError, RuntimeError, ValueError) as e:
                # Weight download/cache failures (OSError incl. URLError) or CUDA/state-dict errors.
                logger.warning("LPIPS failed to bind (%s). Defaulting to pure L1.", e)

    def _perceptual_loss(self, p_img: torch.Tensor, t_img: torch.Tensor, t_idx) -> torch.Tensor:
        """Compute LPIPS perceptual loss scaled by task weights or default coefficient."""
        if self.perc is None:
            return torch.tensor(0.0, device=p_img.device)
        p_scaled = torch.clamp(p_img, 0, 1) * 2.0 - 1.0
        t_scaled = torch.clamp(t_img, 0, 1) * 2.0 - 1.0
        raw_lpips = self.perc(p_scaled, t_scaled).view(-1)
        if t_idx is not None and t_idx.ndim > 0:
            return (raw_lpips * self.task_lpips_weights[t_idx]).mean()
        return 0.025 * raw_lpips.mean()

    def _compute_restoration_loss(self, pred, target, task_idx) -> torch.Tensor:
        """Compute composite L1, cross-entropy routing, and perceptual loss."""
        detect_loss = None
        restored = _restoration_output(pred)
        if restored is not None:
            img_pred = restored["restored_image"]
            img_target = target["image"] if isinstance(target, dict) else target
            bbox_pred = restored.get("detections")
            if bbox_pred is not None and isinstance(bbox_pred, torch.Tensor):
                detect_loss = F.smooth_l1_loss(bbox_pred, torch.zeros_like(bbox_pred))
        elif isinstance(pred, (tuple, list)):
            img_pred = pred[0]
            img_target = target
        else:
            img_pred = pred
            img_target = target

        base_loss = self.criteria.l1(img_pred, img_target)
        batch_sz = img_pred.shape[0] if hasattr(img_pred, "shape") else 1
        device = img_pred.device if hasattr(img_pred, "device") else torch.device("cpu")
        task_tensor = _resolve_task_index_tensor(task_idx, batch_sz, device)

        if isinstance(pred, (tuple, list)) and len(pred) > 1 and task_tensor is not None:
            base_loss = base_loss + 0.1 * self.criteria.ce(pred[1], task_tensor)

        base_loss = base_loss + self._perceptual_loss(img_pred, img_target, task_tensor)

        if detect_loss is not None:
            base_loss = base_loss + 0.1 * detect_loss

        return base_loss

    def forward(self, pred, target, task_idx=None):
        """Dispatch to the loss for ``self.task_type`` and return the scalar loss."""
        if self.task_type in ("forex", "forex_dual") or _is_forex_prediction(pred):
            self.forex_dual.stab = self.stab
            return self.forex_dual(pred, target)

        if self.task_type in ["restoration", "enhancement"]:
            return self._compute_restoration_loss(pred, target, task_idx)

        if self.task_type == "quality":
            return self._compute_quality_loss(pred, target)

        return self._compute_fallback_loss(pred, target)

    def _compute_quality_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-objective distribution, rank, and Spearman loss for quality assessment."""
        p_probs = F.softmax(pred.float() / float(self.stab.get("softmax_temp", 1.0)), dim=-1)
        tgt_f = target.float()
        t_probs = tgt_f / torch.clamp(
            tgt_f.sum(dim=-1, keepdim=True), min=float(self.stab.get("emd_epsilon", 1e-6))
        )
        emd = torch.sum(
            (torch.cumsum(p_probs, dim=-1) - torch.cumsum(t_probs, dim=-1)) ** 2, dim=-1
        ).mean()

        p_eval, t_eval = self._mean_scores(p_probs, t_probs)

        total_loss = float(self.stab.get("emd_weight", 1.0)) * emd

        if self.stab.get("use_soft_spearman", True) and p_eval.size(0) > 1:
            spearman_loss = self.criteria.soft_spearman(p_eval, t_eval)
            total_loss = total_loss + (
                float(self.stab.get("soft_spearman_weight", 0.5)) * spearman_loss
            )

        ssim_w = float(self.stab.get("ssim_weight", 0.0))
        if ssim_w > 0.0 and p_eval.size(0) > 1:
            total_loss = total_loss + ssim_w * _correlation_proxy(p_eval, t_eval)

        rank_weight = float(self.stab.get("rank_weight", 0.0))
        if rank_weight > 0 and p_eval.size(0) > 1:
            rank_loss = _pairwise_rank_loss(
                p_eval, t_eval, float(self.stab.get("rank_margin", 0.05))
            )
            total_loss = total_loss + (rank_weight * rank_loss)

        return total_loss

    def _mean_scores(
        self, p_probs: torch.Tensor, t_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean 1..10 scores; with a memory bank, the bank context (then the bank is updated)."""
        p_mean = (p_probs * self.rank_weights).sum(dim=-1)
        t_mean = (t_probs * self.rank_weights).sum(dim=-1)

        if self.memory_bank is None:
            return p_mean, t_mean
        p_eval, t_eval = self.memory_bank.get_context(p_mean, t_mean)
        self.memory_bank.update(p_mean, t_mean)
        return p_eval, t_eval

    def _compute_fallback_loss(self, pred: Any, target: Any) -> torch.Tensor:
        """Fallback loss computation for detection, segmentation, classification, or generic MSE."""
        if self.task_type == "parameter_prediction":
            pred_norm = pred.clone()
            target_norm = target.clone()
            if pred_norm.shape[-1] >= 3:
                pred_norm[:, 1] = pred_norm[:, 1] / 3.14159265  # theta / pi -> [0,1]
                target_norm[:, 1] = target_norm[:, 1] / 3.14159265
            huber_delta = float(self.stab.get("huber_delta", 1.0))
            return F.smooth_l1_loss(pred_norm, target_norm, beta=huber_delta)

        if self.task_type == "classification":
            return self.criteria.ce(pred, target)

        if self.task_type == "segmentation":
            if target.dim() == 4:
                target = target.squeeze(1)
            elif target.dim() == 2 and target.size(1) == 1:
                target = torch.zeros(
                    (pred.size(0), pred.size(2), pred.size(3)), dtype=torch.long, device=pred.device
                )
            return self.criteria.ce(pred, target.long())

        if (
            self.task_type == "face_detection"
            and isinstance(pred, (tuple, list))
            and len(pred) == 3
        ):
            return _face_detection_loss(pred, target)

        return self.criteria.mse(pred, target)
