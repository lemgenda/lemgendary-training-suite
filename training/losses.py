import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    LemGendary 2026 Unified Loss Engine
    Supports Restoration (L1+LPIPS), Quality (EMD+RankBoost), and Classification (CE).
    """
    def __init__(self, task_type="restoration", stabilizers=None):
        super().__init__()
        self.task_type = task_type
        # 2026 Resilience: Dynamic injection from config hierarchy
        self.stab = stabilizers or {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0}
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean') # Legacy fallback for face and segmentation topology
        self.ce = nn.CrossEntropyLoss()
        self.perc = None

        # 2026: SOTA Rank-Boost Weights (Standard 10..1 mapping)
        self.register_buffer('rank_weights', torch.arange(10, 0, -1).float())

        if self.task_type in ["restoration", "enhancement"]:
            try:
                import lpips
                # 2026: Mission Pulse - Restore transparency for slow perceptual engine loading
                print(" [MISSION] Initializing Neural Perceptual Engine (LPIPS/VGG16)...")
                # Natively trained perceptual alignment! Exponentially more stable than crude VGG L1
                self.perc = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')
                self.perc.eval()
                for param in self.perc.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"⚠️ [RESILIENCE] LPIPS failed to bind ({e}). Defaulting to pure L1.")

    def forward(self, pred, target, task_idx=None):
        if self.task_type in ["restoration", "enhancement"]:
            # Support both Hybrid (img, weights) outputs and Pure Image generation
            if isinstance(pred, (tuple, list)):
                base_loss = self.l1(pred[0], target) + 0.1 * self.ce(pred[1], task_idx)
                if self.perc is not None:
                    # LPIPS natively outputs spatial arrays. Mean() required. Clamp to [-1, 1].
                    p_scaled = torch.clamp(pred[0], 0, 1) * 2.0 - 1.0
                    t_scaled = torch.clamp(target, 0, 1) * 2.0 - 1.0
                    # 2026 Shift: Balanced at 0.025 to create geometric harmony between PSNR & Perception
                    base_loss += 0.025 * self.perc(p_scaled, t_scaled).mean()
                return base_loss
            else:
                base_loss = self.l1(pred, target)
                if self.perc is not None:
                    p_scaled = torch.clamp(pred, 0, 1) * 2.0 - 1.0
                    t_scaled = torch.clamp(target, 0, 1) * 2.0 - 1.0
                    # 2026 Shift: Balanced at 0.025 to natively harmonize metric extraction
                    base_loss += 0.025 * self.perc(p_scaled, t_scaled).mean()
                return base_loss
        
        elif self.task_type == "quality":
            pred_f = pred.float()
            tgt_f = target.float()

            # NIMA specific Earth Mover's Distance (EMD) with sharpened Logit Anchoring
            p_probs = F.softmax(pred_f.clamp(min=-self.stab.get('logit_clamp', 20.0), max=self.stab.get('logit_clamp', 20.0)) / self.stab.get("softmax_temp", 1.0), dim=-1)
            t_probs = tgt_f / torch.clamp(tgt_f.sum(dim=-1, keepdim=True), min=self.stab.get("emd_epsilon", 1e-6))

            cdf_p = torch.cumsum(p_probs, dim=-1)
            cdf_t = torch.cumsum(t_probs, dim=-1)

            # 2026: Geometric Stabilizer - Summing squared CDF error per-bin
            emd = torch.sum((cdf_p - cdf_t) ** 2, dim=-1).mean()

            # --- 2026: Neural Rank-Boost (SRCC Enhancement) ---
            rank_weight = self.stab.get('rank_weight', 0.0)
            if rank_weight > 0 and p_probs.size(0) > 1:
                p_mean = (p_probs * self.rank_weights).sum(dim=-1)
                t_mean = (t_probs * self.rank_weights).sum(dim=-1)

                p_diff = p_mean.unsqueeze(0) - p_mean.unsqueeze(1)
                t_diff = t_mean.unsqueeze(0) - t_mean.unsqueeze(1)
                t_sign = torch.sign(t_diff)

                margin = self.stab.get('rank_margin', 0.05)
                rank_loss = F.relu(margin - t_sign * p_diff)
                mask = (t_sign != 0).float()
                avg_rank_loss = (rank_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                return emd + (rank_weight * avg_rank_loss)

            return emd
            
        elif self.task_type == "classification":
            return self.ce(pred, target)
            
        return self.mse(pred, target)
