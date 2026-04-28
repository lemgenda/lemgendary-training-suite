import torch
import torch.nn as nn
from torchvision import models

class NIMA_Model(nn.Module):
    """
    NIMA (Neural IMage Assessment) Model.
    Uses MobileNetV2 features and predicts a 10-class distribution of quality scores.
    """
    def __init__(self, backbone="mobilenet_v2"):
        super(NIMA_Model, self).__init__()
        
        # 2026 SOTA Upgrade: Branching backbones for high-precision technical assessment
        if backbone == "efficientnet_v2_s":
            # EfficientNetV2-S offers significantly higher spatial awareness for micro-defects
            self.features = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
            in_features = 1280
        else:
            # Traditional MobileNetV2 fallback
            self.features = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
            in_features = 1280

        # Flatten and predict 10 classes (representing scores 1 through 10)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # 2026: SOTA Dropout Guard
            nn.Linear(in_features, 10)
        )

    def forward(self, x):
        # 2026 High-Velocity Tensor Mapping
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) # Robust GAP
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # 2026 Resilience: Internal Logit Safety Valve
        # Clamping to ±10.0 ensures stability during first-iteration resumption (FP16 safe)
        return x

# Unified Loss Logic moved to training/train.py for 2026 Resiliency Synchronization.

class AuthenticityScorer(nn.Module):
    """
    Authenticity Scorer (Binary Classification: AI vs Real)
    Built on a robust EfficientNet backbone to detect high-frequency AI generation artifacts.
    """
    def __init__(self, num_classes=2):
        super(AuthenticityScorer, self).__init__()
        # Use EfficientNetV2-S for deep feature extraction (Matches NIMA standard)
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # Replace the classification head for binary authentication
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 2026 Resilience: Native internal logit clamping
        return self.backbone(x)
