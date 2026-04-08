import torch
import torch.nn as nn

class UNetProxy(nn.Module):
    def __init__(self, in_c, out_c, channels=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, channels, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(channels, channels*2, 3, 2, 1), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(channels*2, channels*2, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels*2, channels, 4, 2, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(channels*2, channels, 4, 2, 1), nn.ReLU())
        self.out = nn.Conv2d(channels + in_c, out_c, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        m = self.mid(e2)
        d2 = self.dec2(m)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        out = self.out(torch.cat([d1, x], dim=1))
        return out

class CodeFormerMock(nn.Module): # Kept name for factory routing
    """ 
    Structurally sound UNet proxy for Face Restoration.
    """
    def __init__(self):
        super().__init__()
        self.net = UNetProxy(3, 3, 64)
    
    def forward(self, x): 
        return self.net(x)

class ParseNetMock(nn.Module): # Kept name for factory routing
    """ 
    Structurally sound UNet proxy for Face Parsing Segmentation.
    """
    def __init__(self):
        super().__init__()
        self.net = UNetProxy(3, 19, 32)
    
    def forward(self, x): 
        return self.net(x)
