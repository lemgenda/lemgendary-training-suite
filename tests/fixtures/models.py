"""Tiny torch models shared by the hardware, VRAM and governor tests."""

import torch
from torch import nn


class SingleConvModel(nn.Module):
    """Single 3x3 conv restoration model used as a hardware-audit and governor fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply the conv to ``x`` (or to ``x["pixel_values"]`` for dict batches)."""
        if isinstance(x, dict):
            x = x["pixel_values"]
        return self.conv(x)
