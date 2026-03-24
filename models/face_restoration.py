import torch
import torch.nn as nn

class CodeFormerMock(nn.Module):
    """ 
    Mocks CodeFormer image-to-image autoencoder (512x512). 
    Used for face restoration and enhancement.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid()
        )
    
    def forward(self, x): 
        return self.net(x)

class ParseNetMock(nn.Module):
    """ 
    Mocks BiSeNet face parsing (512x512 -> 19 classes). 
    Used for semantic segmentation of face components.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 19, 3, padding=1)
        )
    
    def forward(self, x): 
        return self.net(x)
