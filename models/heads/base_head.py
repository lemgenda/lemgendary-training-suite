"""
Base convolutional restoration head for multitask vision restoration models.
"""

import torch.nn as nn


class BaseRestorationHead(nn.Module):
    """Base convolutional output head for multi-task restoration models."""

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
