import torch
import torch.nn as nn
from torchvision import models

class NIMA_Model(nn.Module):
    """
    NIMA (Neural IMage Assessment) Model.
    Uses MobileNetV2 features and predicts a 10-class distribution of quality scores.
    """
    def __init__(self, base_model="mobilenet_v2"):
        super(NIMA_Model, self).__init__()
        # Load a pretrained feature extractor
        self.features = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features

        # Flatten and predict 10 classes (representing scores 1 through 10)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3]) # Global Average Pooling
        return self.classifier(x)

def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance (EMD) Loss formulation.
    p: predicted distribution (batch_size, 10)
    q: ground truth distribution (batch_size, 10)
    """
    cdf_p = torch.cumsum(p, dim=1)
    cdf_q = torch.cumsum(q, dim=1)
    cdf_diff = torch.abs(cdf_p - cdf_q) ** r
    return torch.mean(torch.sum(cdf_diff, dim=1))
