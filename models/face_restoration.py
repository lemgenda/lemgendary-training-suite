import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual

class UNetBackbone(nn.Module):
    """
    Real High-Fidelity UNet Backbone with Residual Blocks.
    Designed for Face Restoration and Parsing.
    """
    def __init__(self, in_c, out_c, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_c, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        
        # Middle
        self.mid = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            ResidualBlock(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(base_channels, out_c, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        
        m = self.mid(self.down2(e2))
        
        d2 = self.dec2(self.up2(m) + e2) # Skip connection
        d1 = self.dec1(self.up1(d2) + e1) # Skip connection
        
        return self.final(d1)

class CodeFormer(nn.Module):
    """
    Real CodeFormer-Style Face Restoration Model.
    Uses a Deep Residual UNet for high-fidelity reconstruction.
    """
    def __init__(self):
        super().__init__()
        self.unet = UNetBackbone(3, 3, 64)
        
    def forward(self, x):
        # In CodeFormer, the output is often a residual of the input
        return x + self.unet(x)

class ParseNet(nn.Module):
    """
    Real ParseNet Face Parsing Model.
    Outputs a segmentation map with 19 classes.
    """
    def __init__(self):
        super().__init__()
        # 19 Classes for face parsing
        self.unet = UNetBackbone(3, 19, 32)
        
    def forward(self, x):
        return self.unet(x)
