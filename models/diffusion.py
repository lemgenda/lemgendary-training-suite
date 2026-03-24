import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    Minimal UNet for Diffusion-based Refinement.
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 2, 1)
        )
        
        self.time_embed = nn.Embedding(1000, base_ch)
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t).view(-1, 64, 1, 1)
        
        h = self.down(x)
        h = h + t_emb
        return self.up(h)

class DiffusionRefiner:
    """
    Helper for Diffusion-based image refinement.
    """
    def __init__(self, device='cpu'):
        self.model = SimpleUNet().to(device)
        self.device = device
        
    def train_step(self, img, optimizer):
        B = img.shape[0]
        noise = torch.randn_like(img).to(self.device)
        t = torch.randint(0, 1000, (B,), device=self.device)
        
        # Add noise
        noisy = img + noise * (t.view(-1, 1, 1, 1).float() / 1000.0)
        
        pred_noise = self.model(noisy, t)
        loss = F.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
