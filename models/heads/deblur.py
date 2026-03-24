import torch.nn as nn

class DeblurHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
