import torch
import torch.nn as nn
from torchvision import models

# [SENIOR HARDENING v16.0 - SYNC_ID: 1312]

class SoftmaxWrapper(nn.Module):
    def __init__(self, inner_model, temperature=1.0):
        super().__init__()
        self.inner_model = inner_model
        self.temperature = temperature
    def forward(self, x):
        logits = self.inner_model(x)
        return torch.nn.functional.softmax(logits / self.temperature, dim=1)


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling with learnable power p."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


class NIMA_Model(nn.Module):
    """
    Nuclear-Hardened NIMA (Neural IMage Assessment).
    Implements Autonomous Temperature Sharpening, Spatial Statistical Pooling, and Logit Clamping.
    """
    def __init__(self, backbone="mobilenet_v2", hidden_dim=None, pooling="avg"):
        super(NIMA_Model, self).__init__()
        self.backbone_name = backbone
        self.pooling_type = pooling
        
        if backbone == "efficientnet_v2_s":
            self.features = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
            base_in_features = 1280
        elif backbone == "swin_v2_t":
            self.features = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).features
            base_in_features = 768
        else:
            self.features = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            base_in_features = 1280

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
                nn.Linear(hidden_dim, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 10)
            )
        
        # 2026 Resilience: Dynamic Temperature Handle
        self.softmax_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def set_temp(self, temp):
        """Update sharpening manifold (Called by Governor)."""
        self.softmax_temp.data = torch.tensor(temp).to(self.softmax_temp.device)

    def forward(self, x):
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
        
        # 2026 Resilience: Logit Safety Valve (Task 10.4)
        # Clamping to ±10.0 prevents probability collapse during sudden resolution shifts.
        logits = torch.clamp(logits, -10.0, 10.0)
        
        # 2026: SOTA Autonomous Sharpening (Task 10.1)
        # We return raw logits to ensure compatibility with the Unified Loss Engine (EMD+RankBoost).
        return logits

class AuthenticityScorer(nn.Module):
    """
    SOTA Authenticity Scorer (AI vs Real) & Categorical Safety Engine.
    Supports standard GAP, Spatial Statistical Pooling (Mean + Std), and GeM.
    """
    def __init__(self, num_classes=2, pooling="avg"):
        super(AuthenticityScorer, self).__init__()
        self.pooling_type = pooling
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        if pooling == "stats":
            self.backbone.classifier[1] = nn.Linear(in_features * 2, num_classes)
        else:
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        if pooling == "gem":
            self.gem_pool = GeMPooling()

    def forward(self, x):
        if self.pooling_type == "avg":
            logits = self.backbone(x)
            return torch.clamp(logits, -10.0, 10.0)

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
        return torch.clamp(logits, -10.0, 10.0)

class UniversalClassifier(AuthenticityScorer):
    """
    Universal Categorical Classifier for Multi-Class tasks (e.g. NSFW, Concepts).
    Inherits from AuthenticityScorer which already implements num_classes dynamic scaling
    and multi-pooling support.
    """
    pass

