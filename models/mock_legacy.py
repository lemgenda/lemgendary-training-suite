import torch.nn as nn

class GenericRestorationMock(nn.Module):
    """
    Mock model for standard image restoration tasks like Dehazing, Deraining, LowLight (MIRNet, FFANet, MPRNet, NAFNet).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        
    def forward(self, x):
        return x + self.net(x)

class UltraZoomModel(nn.Module):
    """
    Mock model for Super Resolution.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3 * (scale_factor ** 2), 3, padding=1)
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        feat = self.net(x)
        return self.pixel_shuffle(feat)

class UniversalFilmRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class UPN_v2_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, 3) # deg, theta, conf
    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1).squeeze(-1))
