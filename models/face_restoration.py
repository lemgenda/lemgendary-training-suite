"""Face restoration (CodeFormer-style) and face parsing (ParseNet) models."""

from torch import nn


class ResidualBlock(nn.Module):
    """Conv-InstanceNorm residual block."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        """Return ``x`` plus the two-conv residual branch."""
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual


def _unet_layers(in_c: int, out_c: int, base_channels: int) -> dict[str, nn.Module]:
    """UNetBackbone children in registration order (the names are the ``state_dict`` keys)."""
    c1, c2, c4 = base_channels, base_channels * 2, base_channels * 4
    return {
        # Encoder
        "enc1": nn.Sequential(
            nn.Conv2d(in_c, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
        ),
        "down1": nn.Conv2d(c1, c2, 4, stride=2, padding=1),
        "enc2": nn.Sequential(ResidualBlock(c2), nn.ReLU(inplace=True)),
        "down2": nn.Conv2d(c2, c4, 4, stride=2, padding=1),
        # Middle
        "mid": nn.Sequential(ResidualBlock(c4), ResidualBlock(c4)),
        # Decoder
        "up2": nn.ConvTranspose2d(c4, c2, 4, stride=2, padding=1),
        "dec2": nn.Sequential(ResidualBlock(c2), nn.ReLU(inplace=True)),
        "up1": nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
        "dec1": nn.Sequential(ResidualBlock(c1), nn.ReLU(inplace=True)),
        "final": nn.Conv2d(c1, out_c, 3, padding=1),
    }


class UNetBackbone(nn.Module):
    """
    Real High-Fidelity UNet Backbone with Residual Blocks.
    Designed for Face Restoration and Parsing.

    Children are registered by name (see ``_unet_layers``) so the checkpoint
    ``state_dict`` keys stay ``enc1.0.weight``, ``down1.weight``...
    """

    enc1: nn.Sequential
    down1: nn.Conv2d
    enc2: nn.Sequential
    down2: nn.Conv2d
    mid: nn.Sequential
    up2: nn.ConvTranspose2d
    dec2: nn.Sequential
    up1: nn.ConvTranspose2d
    dec1: nn.Sequential
    final: nn.Conv2d

    def __init__(self, in_c, out_c, base_channels=64):
        super().__init__()
        for name, layer in _unet_layers(in_c, out_c, base_channels).items():
            self.add_module(name, layer)

    def forward(self, x):
        """Run the UNet encoder/decoder with skip connections."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))

        m = self.mid(self.down2(e2))

        d2 = self.dec2(self.up2(m) + e2)  # Skip connection
        d1 = self.dec1(self.up1(d2) + e1)  # Skip connection

        return self.final(d1)


class CodeFormer(nn.Module):
    """
    Real CodeFormer-Style Face Restoration Model.
    Uses a Deep Residual UNet for high-fidelity reconstruction.
    """

    head_param_prefixes: tuple[str, ...] = ("unet.final.",)
    """Output convolution of the restoration U-Net."""

    def __init__(self, **kwargs):
        super().__init__()
        self.unet = UNetBackbone(3, 3, 64)

    def forward(self, x):
        """Return the input plus the UNet-predicted residual."""
        # In CodeFormer, the output is often a residual of the input
        return x + self.unet(x)


class ParseNet(nn.Module):
    """
    Real ParseNet Face Parsing Model.
    Outputs a segmentation map with 19 classes.
    """

    head_param_prefixes: tuple[str, ...] = ("unet.final.",)
    """Per-pixel class projection of the parsing U-Net."""

    def __init__(self, **kwargs):
        super().__init__()
        # 19 Classes for face parsing
        self.unet = UNetBackbone(3, 19, 32)

    def forward(self, x):
        """Return per-pixel logits for the 19 face-parsing classes."""
        return self.unet(x)
