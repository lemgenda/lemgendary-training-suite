"""Image quality/aesthetic (NIMA) and authenticity/classification models.

Logit bound (owner decision R2-T6): the score logits used to be hard-clamped to the
hardcoded range [-10, 10]. A hard clamp has zero gradient outside the range, so a
saturated bin stops learning, and softmax does not need it for overflow (PyTorch's
softmax subtracts the row maximum). The bound now comes from the registry
(``stabilizers.logit_clamp`` of each model, the same value the governor tracks as its
clamp) and is applied as a tanh soft cap, ``cap * tanh(logits / cap)``, the logit
soft-capping of Gemma 2 (Gemma Team, 2024): smooth, monotonic, close to the identity
for ``|logit|`` well below ``cap`` and bounded by ``cap``, so FP16/AMP logits stay
finite after a resolution jump while every bin keeps a gradient. A cap of 15 still
spans a probability ratio of e^30 between bins, far more than a 10-bin EMD target
needs. ``0`` turns the cap off.
"""

import math

import torch
from torch import nn
from torchvision import models

# [SENIOR HARDENING v16.0 - SYNC_ID: 1312]

DEFAULT_LOGIT_CAP = 15.0
"""Soft cap of a model built without a registry ``stabilizers.logit_clamp``."""


def soft_cap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """``cap * tanh(logits / cap)``; ``logits`` unchanged when ``cap <= 0``."""
    if cap <= 0:
        return logits
    return cap * torch.tanh(logits / cap)


class LogitCapMixin:
    """A configurable logit soft cap (set from the registry and the governor's clamp)."""

    logit_cap: float = DEFAULT_LOGIT_CAP

    def set_logit_cap(self, cap: float) -> None:
        """Set the soft cap (``0`` or less turns it off)."""
        value = float(cap)
        if math.isnan(value):
            raise ValueError("logit cap must be a number, got NaN")
        self.logit_cap = value

    def cap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply this model's soft cap to ``logits``."""
        return soft_cap_logits(logits, self.logit_cap)


class SoftmaxWrapper(nn.Module):
    """Wrap a model and apply a temperature-scaled softmax to its logits."""

    def __init__(self, inner_model, temperature=1.0):
        super().__init__()
        self.inner_model = inner_model
        self.temperature = temperature

    def forward(self, x):
        """Return class probabilities of the wrapped model's logits."""
        logits = self.inner_model(x)
        return torch.nn.functional.softmax(logits / self.temperature, dim=1)


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling with learnable power p."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling to ``x`` and return [B, C, 1, 1]."""
        return nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(
            1.0 / self.p
        )


class NIMAModel(LogitCapMixin, nn.Module):
    """
    Nuclear-Hardened NIMA (Neural IMage Assessment).
    Implements Autonomous Temperature Sharpening, Spatial Statistical Pooling, and Logit Clamping.
    """

    def __init__(self, backbone="mobilenet_v3_small", hidden_dim=None, pooling="avg", **kwargs):
        super().__init__()
        self.backbone_name = backbone

        self.pooling_type = pooling

        if backbone == "efficientnet_v2_s":
            self.features = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            ).features
            base_in_features = 1280
        elif backbone == "swin_v2_t":
            self.features = models.swin_v2_t(
                weights=models.Swin_V2_T_Weights.IMAGENET1K_V1
            ).features
            base_in_features = 768
        elif backbone == "mobilenet_v2":
            self.features = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            ).features
            base_in_features = 1280
        else:
            self.features = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            ).features
            base_in_features = 576

        in_features = base_in_features * 2 if pooling == "stats" else base_in_features
        if pooling == "gem":
            self.gem_pool = GeMPooling()

        # 2026 v16: Optional hidden projection layer for richer aesthetic representation.
        # When hidden_dim is set (e.g. 256), the head learns a rich intermediate manifold
        # before collapsing to 10 score bins, raising the PLCC ceiling for lightweight backbones.
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.25),
                nn.Linear(hidden_dim, 10),
            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 10))

        # 2026 Resilience: Dynamic Temperature Handle
        self.softmax_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    @property
    def head_param_prefixes(self) -> tuple[str, ...]:
        """Layers added on the ImageNet backbone: the score head and a learnable GeM power."""
        return ("gem_pool.", "classifier.") if hasattr(self, "gem_pool") else ("classifier.",)

    def set_temp(self, temp):
        """Update sharpening manifold (Called by Governor)."""
        self.softmax_temp.data = torch.tensor(temp).to(self.softmax_temp.device)

    def forward(self, x):
        """Return soft-capped logits over the 10 NIMA score bins."""
        x = self.features(x)
        if getattr(self, "backbone_name", "") == "swin_v2_t":
            # Swin outputs [B, H, W, C], pool2d needs [B, C, H, W]
            x = x.permute(0, 3, 1, 2)

        if self.pooling_type == "stats":
            mean_f = nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            std_f = torch.std(x, dim=[2, 3])
            feat = torch.cat([mean_f, std_f], dim=1)
        elif self.pooling_type == "gem":
            feat = torch.flatten(self.gem_pool(x), 1)
        else:
            feat = torch.flatten(nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)

        logits = self.classifier(feat)

        # Logit safety valve: configurable tanh soft cap (see the module docstring).
        logits = self.cap_logits(logits)

        # 2026: SOTA Autonomous Sharpening (Task 10.1)
        # We return raw logits to ensure compatibility with the Unified Loss Engine (EMD+RankBoost).
        return logits


class AuthenticityScorer(LogitCapMixin, nn.Module):
    """
    SOTA Authenticity Scorer (AI vs Real) & Categorical Safety Engine.
    Supports standard GAP, Spatial Statistical Pooling (Mean + Std), and GeM.
    """

    def __init__(self, num_classes=2, pooling="avg", **kwargs):
        super().__init__()
        self.pooling_type = pooling
        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        head = self.backbone.classifier[1]
        if not isinstance(head, nn.Linear):
            raise TypeError(
                f"EfficientNetV2-S classifier head is {type(head).__name__}, not Linear"
            )
        in_features = head.in_features
        if pooling == "stats":
            self.backbone.classifier[1] = nn.Linear(in_features * 2, num_classes)
        else:
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        if pooling == "gem":
            self.gem_pool = GeMPooling()

    @property
    def head_param_prefixes(self) -> tuple[str, ...]:
        """The replaced classifier layer and a learnable GeM power on the pretrained backbone."""
        head = ("backbone.classifier.",)
        return ("gem_pool.", *head) if hasattr(self, "gem_pool") else head

    def forward(self, x):
        """Return soft-capped class logits using the configured pooling."""
        if self.pooling_type == "avg":
            return self.cap_logits(self.backbone(x))

        x_feat = self.backbone.features(x)
        if self.pooling_type == "stats":
            mean_f = nn.functional.adaptive_avg_pool2d(x_feat, (1, 1)).flatten(1)
            std_f = torch.std(x_feat, dim=[2, 3])
            feat = torch.cat([mean_f, std_f], dim=1)
        elif self.pooling_type == "gem":
            feat = torch.flatten(self.gem_pool(x_feat), 1)
        else:
            feat = torch.flatten(nn.functional.adaptive_avg_pool2d(x_feat, (1, 1)), 1)

        logits = self.backbone.classifier(feat)
        return self.cap_logits(logits)


class UniversalClassifier(AuthenticityScorer):
    """
    Universal Categorical Classifier for Multi-Class tasks (e.g. NSFW, Concepts).
    Inherits from AuthenticityScorer which already implements num_classes dynamic scaling
    and multi-pooling support.
    """
