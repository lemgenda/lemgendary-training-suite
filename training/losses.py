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

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))

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
