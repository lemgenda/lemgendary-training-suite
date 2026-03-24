import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)

        # Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        return x_flat.permute(0, 2, 1).view(B, C, H, W)

class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=3, dim=64, num_blocks=6):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, dim, 3, 1, 1)

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.conv(x)
        return self.blocks(x)

# Rename SharedEncoder to maintain compatibility or just alias it
SharedEncoder = TransformerEncoder
