import torch.nn as nn

class DenoiseHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
