import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """
    LemGendary 2026 Unified Loss Engine
    Supports Restoration (L1+LPIPS), Quality (EMD+RankBoost), and Classification (CE).
    """
    def __init__(self, task_type="restoration", stabilizers=None, use_perc=False):
        super().__init__()
        self.task_type = task_type
        # 2026 Resilience: Dynamic injection from config hierarchy
        self.stab = stabilizers or {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0}
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean') # Legacy fallback for face and segmentation topology
        self.ce = nn.CrossEntropyLoss()
        self.perc = None

        # 2026: SOTA Rank-Boost Weights (Standard 10..1 mapping)
        self.register_buffer('rank_weights', torch.arange(1, 11).float())

        # 2026: MoE 11-Manifold Perceptual Scaling Coefficients
        # Balances LPIPS gradients to prevent high-frequency tasks (Face, SuperRes) from
        # overwhelming intensity-based tasks (Denoise, Lowlight).
        self.register_buffer('task_lpips_weights', torch.tensor([
            0.010, # 0: denoise
            0.025, # 1: deblur
            0.020, # 2: derain
            0.015, # 3: dehaze_indoor
            0.015, # 4: dehaze_outdoor
            0.010, # 5: lowlight
            0.010, # 6: exposure
            0.030, # 7: superres
            0.050, # 8: vintage
            0.050, # 9: face_restorer
            0.005  # 10: face_parser
        ], dtype=torch.float32))

        if self.task_type in ["restoration", "enhancement"] and use_perc:
            try:
                # 2026 Resilience: Surgical Memory Reclamation before heavy Perceptual Engine load
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                import lpips
                # 2026: Mission Pulse - Restore transparency for slow perceptual engine loading
                print(" [MISSION] Initializing Neural Perceptual Engine (LPIPS/VGG16)...")
                # Natively trained perceptual alignment! Exponentially more stable than crude VGG L1
                self.perc = lpips.LPIPS(net='vgg')
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
                    # LPIPS natively outputs spatial arrays. Clamp to [-1, 1].
                    p_scaled = torch.clamp(pred[0], 0, 1) * 2.0 - 1.0
                    t_scaled = torch.clamp(target, 0, 1) * 2.0 - 1.0
                    
                    raw_lpips = self.perc(p_scaled, t_scaled).view(-1) # [B]
                    
                    if task_idx is not None and len(task_idx.shape) > 0:
                        # Extract the dynamic weight for each sample in the batch
                        dynamic_weights = self.task_lpips_weights[task_idx]
                        perc_loss = (raw_lpips * dynamic_weights).mean()
                    else:
                        perc_loss = 0.025 * raw_lpips.mean()
                        
                    base_loss += perc_loss
                return base_loss
            else:
                base_loss = self.l1(pred, target)
                if self.perc is not None:
                    p_scaled = torch.clamp(pred, 0, 1) * 2.0 - 1.0
                    t_scaled = torch.clamp(target, 0, 1) * 2.0 - 1.0
                    
                    raw_lpips = self.perc(p_scaled, t_scaled).view(-1) # [B]
                    
                    if task_idx is not None and len(task_idx.shape) > 0:
                        dynamic_weights = self.task_lpips_weights[task_idx]
                        perc_loss = (raw_lpips * dynamic_weights).mean()
                    else:
                        perc_loss = 0.025 * raw_lpips.mean()
                        
                    base_loss += perc_loss
                return base_loss
        
        elif self.task_type == "quality":
            pred_f = pred.float()
            tgt_f = target.float()

            # NIMA specific Earth Mover's Distance (EMD) with sharpened Logit Anchoring
            p_probs = F.softmax(pred_f.clamp(min=-float(self.stab.get('logit_clamp', 20.0)), max=float(self.stab.get('logit_clamp', 20.0))) / float(self.stab.get("softmax_temp", 1.0)), dim=-1)
            t_probs = tgt_f / torch.clamp(tgt_f.sum(dim=-1, keepdim=True), min=float(self.stab.get("emd_epsilon", 1e-6)))

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
            
        elif self.task_type == "parameter_prediction":
            # 2026: Bounded Regression Loss for UPNv2 Parameter Predictor
            # SmoothL1 (Huber) is robust to outliers and stable for bounded [0,1] targets.
            # Normalize theta by π so all parameters contribute equally in [0,1] range.
            pred_norm = pred.clone()
            target_norm = target.clone()
            if pred_norm.shape[-1] >= 3:
                pred_norm[:, 1] = pred_norm[:, 1] / 3.14159265  # theta / π → [0,1]
                target_norm[:, 1] = target_norm[:, 1] / 3.14159265
            return F.smooth_l1_loss(pred_norm, target_norm)

        elif self.task_type == "classification":
            return self.ce(pred, target)
            
        elif self.task_type == "segmentation":
            if target.dim() == 4:
                target = target.squeeze(1)
            elif target.dim() == 2 and target.size(1) == 1:
                target = torch.zeros((pred.size(0), pred.size(2), pred.size(3)), dtype=torch.long, device=pred.device)
            return self.ce(pred, target.long())

        elif self.task_type == "face_detection":
            if isinstance(pred, (tuple, list)) and len(pred) == 3:
                p_bbox, p_conf, p_landm = pred
                t_conf = target[:, 0:1]
                t_bbox = target[:, 1:5]
                t_landm = target[:, 5:15]
                
                loss_conf = F.binary_cross_entropy(p_conf, t_conf)
                
                pos_mask = t_conf > 0.5
                if pos_mask.sum() > 0:
                    loss_bbox = F.smooth_l1_loss(p_bbox[pos_mask.squeeze(-1)], t_bbox[pos_mask.squeeze(-1)])
                    loss_landm = F.smooth_l1_loss(p_landm[pos_mask.squeeze(-1)], t_landm[pos_mask.squeeze(-1)])
                else:
                    loss_bbox = torch.tensor(0.0, device=p_bbox.device)
                    loss_landm = torch.tensor(0.0, device=p_landm.device)
                    
                return loss_conf + loss_bbox * 2.0 + loss_landm * 1.0
            return self.mse(pred, target)
            
        return self.mse(pred, target)
