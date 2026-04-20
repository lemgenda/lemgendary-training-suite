import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 64, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x

class GenericRestorationModel(nn.Module):
    """
    Real High-Fidelity Residual Restoration Model.
    Replaces GenericRestorationMock.
    """
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.body = nn.Sequential(
            ResidualDenseBlock(32),
            ResidualDenseBlock(32)
        )
        self.outro = nn.Conv2d(32, 3, 3, padding=1)
        
    def forward(self, x):
        feat = self.intro(x)
        out = self.body(feat)
        return x + self.outro(out)

class UltraZoomModel(nn.Module):
    """
    Real Super-Resolution Model (ESPCN based).
    Replaces UltraZoomMock.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.intro = nn.Conv2d(3, 64, 5, padding=2)
        self.body = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * (scale_factor ** 2), 3, padding=1)
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        feat = self.body(self.intro(x))
        return self.pixel_shuffle(feat)

class UniversalFilmRestorer(nn.Module):
    """Autoencoder-style film restorer using Residual Dense Blocks."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(64)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class UPN_v2_Model(nn.Module):
    """Universal Parameter Predictor with MobileNet-lite backbone."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 3) # deg, theta, conf
    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.fc(feat)
