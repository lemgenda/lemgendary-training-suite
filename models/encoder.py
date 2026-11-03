"""Shared transformer encoder used by the multi-task restorer."""

import torch.nn.functional as F
from torch import nn


class TransformerBlock(nn.Module):
    """Transformer block applying self-attention on a 32x32 pooled feature map."""

    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        """Attend over pooled features, upsample back and add to ``x``."""
        batch, channels, height, width = x.shape

        # 2026 Resilience: Downsample sequence length to prevent quadratic spatial complexity OOM
        # Downsample to 32x32 features for efficient global contextual attention
        target_size = (32, 32)
        x_down = F.adaptive_avg_pool2d(x, target_size)

        x_flat = x_down.view(batch, channels, target_size[0] * target_size[1]).permute(0, 2, 1)

        # Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        # Restore resolution
        x_out = x_flat.permute(0, 2, 1).view(batch, channels, target_size[0], target_size[1])
        x_up = F.interpolate(x_out, size=(height, width), mode="bilinear", align_corners=False)

        # Residual connection at original scale
        return x + x_up


class TransformerEncoder(nn.Module):
    """Conv stem followed by a stack of transformer blocks."""

    def __init__(self, in_ch=3, dim=64, num_blocks=6):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.blocks = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks)])

    def forward(self, x):
        """Embed ``x`` and run it through the transformer blocks."""
        x = self.conv(x)
        return self.blocks(x)


# Rename SharedEncoder to maintain compatibility or just alias it
SharedEncoder = TransformerEncoder
