import torch
import torch.nn as nn
from models.encoder import SharedEncoder

class NIMA_Model(nn.Module):
    """
    NIMA: Neural Image Assessment.
    Professional implementation for Aesthetic/Technical scoring.
    """
    def __init__(self, base_channels=32):
        super(NIMA_Model, self).__init__()
        # Reuse the shared encoder as a backbone
        self.encoder = SharedEncoder(base_channels=base_channels)
        
        # NIMA typically uses a pooling + dropout + linear to 10 Softmax bins
        # For simplicity/WebGPU speed, we use a regressor.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, 1),
            nn.Sigmoid() # Normalize to [0, 1] range for generic scoring
        )

    def forward(self, x):
        feat = self.encoder(x)
        score = self.head(feat)
        return score * 10.0 # Scale to 1-10 range
