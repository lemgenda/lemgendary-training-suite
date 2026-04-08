import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        # 2026 Resilience: Strict ImageNet Normalization Anchor
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Dynamically scale [0,1] matrices into pure VGG geometry
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        return F.l1_loss(self.vgg(x_norm), self.vgg(y_norm))

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()

    def forward(self, pred, target, task='denoise'):
        # In a real production setup, task-specific loss weights could be added here
        return (
            self.l1(pred, target)
            + 0.1 * self.perc(pred, target)
        )
