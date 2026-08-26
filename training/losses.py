import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Forex Dual Loss
# ─────────────────────────────────────────────────────────────────────────────

class ForexDualLoss(nn.Module):
    """
    LemGendary Forex Dual Loss Engine.

    Combines:
        - CrossEntropy for direction (Down / Sideways / Up)
        - Huber (SmoothL1) for magnitude (TP pips, SL pips)

    Direction confidence gates magnitude loss:
        Low-confidence bars (high entropy in direction logits) contribute
        proportionally less to the magnitude regression signal, preventing
        the magnitude head from fitting noise on ambiguous bars.

    Args:
        direction_weight: Weight for CE loss component (default 1.0).
        magnitude_weight: Weight for Huber loss component (default 0.5).
        huber_delta:      Huber delta — transitions L2→L1 at this pip threshold.
    """
    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        huber_delta: float = 20.0,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.huber_delta      = huber_delta
        self.ce               = nn.CrossEntropyLoss()

    def forward(self, pred: dict, labels: dict) -> torch.Tensor:
        """
        Args:
            pred:   Dict with 'direction_logits' [B, 3] and 'magnitude' [B, 2]
            labels: Dict with 'direction' [B] (long) and 'magnitude' [B, 2] (float)

        Returns:
            Combined scalar loss.
        """
        dir_logits  = pred["direction_logits"] if "direction_logits" in pred else pred["direction"]  # [B, 3]
        mag_pred    = pred["magnitude"]              # [B, 2]
        dir_target  = labels["direction"]            # [B] long
        mag_target  = labels["magnitude"]            # [B, 2] float

        # Direction loss
        dir_loss = self.ce(dir_logits, dir_target)

        # Confidence gate: high-entropy bars get lower magnitude weight
        with torch.no_grad():
            probs     = torch.softmax(dir_logits, dim=-1)           # [B, 3]
            entropy   = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B]
            max_ent   = torch.log(torch.tensor(3.0, device=probs.device))
            conf_gate = 1.0 - (entropy / max_ent).clamp(0.0, 1.0)  # [B] ∈ [0,1]

        # Huber loss for magnitude, gated by direction confidence
        huber = F.smooth_l1_loss(mag_pred, mag_target, reduction='none', beta=self.huber_delta)  # [B, 2]
        mag_loss = (huber.mean(dim=1) * conf_gate).mean()

        return self.direction_weight * dir_loss + self.magnitude_weight * mag_loss


class SoftSpearmanLoss(nn.Module):
    """
    Differentiable Soft-Spearman Rank Correlation Loss.
    Approximates the rank operator using temperature-scaled sigmoid pairwise comparisons:
        r_i = 1 + \\sum_{j \\ne i} \\sigma((p_i - p_j) / \\tau)
    Optimizes Spearman rank correlation directly with smooth gradients.
    """
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, p_scores: torch.Tensor, t_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_scores: Predicted 1D continuous scores [N]
            t_scores: Target 1D continuous scores [N]
        Returns:
            Scalar loss: 1 - soft_spearman_correlation
        """
        n = p_scores.size(0)
        if n < 2:
            return torch.tensor(0.0, device=p_scores.device, requires_grad=True)

        # Pairwise difference matrices: [N, N]
        p_diff = (p_scores.unsqueeze(0) - p_scores.unsqueeze(1)) / self.temperature
        t_diff = (t_scores.unsqueeze(0) - t_scores.unsqueeze(1)) / self.temperature

        # Soft ranks via sigmoid
        p_ranks = 1.0 + torch.sigmoid(p_diff).sum(dim=1)
        t_ranks = 1.0 + torch.sigmoid(t_diff).sum(dim=1)

        # Center ranks
        p_ranks_c = p_ranks - p_ranks.mean()
        t_ranks_c = t_ranks - t_ranks.mean()

        # Pearson correlation of soft ranks
        cov = (p_ranks_c * t_ranks_c).sum()
        var_p = (p_ranks_c ** 2).sum()
        var_t = (t_ranks_c ** 2).sum()

        denom = torch.sqrt(torch.clamp(var_p * var_t, min=self.eps))
        soft_spearman = cov / denom

        return 1.0 - soft_spearman


class RankMemoryBank:
    """
    Cross-Microbatch FIFO Memory Bank for Rank & Contrastive Supervision.
    Caches detached predictions and targets across gradient accumulation steps,
    allowing effective pairwise ranking over (N_batch + N_memory) samples
    even when local VRAM forces micro-batch size b=2.
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.p_memory = []
        self.t_memory = []

    def get_context(self, p_active: torch.Tensor, t_active: torch.Tensor):
        """
        Concatenates active batch with detached historical representations.
        Gradients backpropagate exclusively through p_active.
        """
        if len(self.p_memory) == 0:
            return p_active, t_active

        p_hist = torch.stack(self.p_memory).to(p_active.device)
        t_hist = torch.stack(self.t_memory).to(t_active.device)

        p_full = torch.cat([p_active, p_hist], dim=0)
        t_full = torch.cat([t_active, t_hist], dim=0)
        return p_full, t_full

    def update(self, p_active: torch.Tensor, t_active: torch.Tensor):
        """Push active samples into detached FIFO queue."""
        with torch.no_grad():
            for p, t in zip(p_active.detach().view(-1), t_active.detach().view(-1)):
                self.p_memory.append(p)
                self.t_memory.append(t)
                if len(self.p_memory) > self.capacity:
                    self.p_memory.pop(0)
                    self.t_memory.pop(0)

    def reset(self):
        """Clear memory at epoch boundary or resolution jump."""
        self.p_memory.clear()
        self.t_memory.clear()


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for Safety / NSFW Categorical Classification.
    Down-weights easy well-classified negatives to focus on hard boundary triggers.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    LemGendary 2026 Unified Loss Engine
    Supports Restoration (L1+LPIPS), Quality (EMD+SoftSpearman+RankBoost), and Classification (CE/Focal).
    """
    def __init__(self, task_type="restoration", stabilizers=None, use_perc=False):
        super().__init__()
        self.task_type = task_type
        # 2026 Resilience: Dynamic injection from config hierarchy
        self.stab = stabilizers or {"softmax_temp": 0.1, "emd_epsilon": 1e-6, "logit_clamp": 15.0}
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean') # Legacy fallback for face and segmentation topology
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.focal = FocalLoss(gamma=2.0, label_smoothing=0.05)
        self.soft_spearman = SoftSpearmanLoss(temperature=0.1)
        self.memory_bank = None
        if self.stab.get("rank_memory_bank_size", 0) > 0 or self.stab.get("use_memory_bank", False):
            mb_size = int(self.stab.get("rank_memory_bank_size", 32))
            self.memory_bank = RankMemoryBank(capacity=mb_size)
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

                # 2026 Resilience: Native fix for torchvision 'pretrained' deprecation warning (Root Cause)
                # Using getattr/setattr bypasses Pyre static type checking errors in your IDE.
                import torchvision.models
                if not hasattr(torchvision.models, '_patched_vgg16_applied'):
                    orig_vgg16 = getattr(torchvision.models, 'vgg16')
                    def patched_vgg16(*args, **kwargs):
                        if 'pretrained' in kwargs:
                            if kwargs.pop('pretrained'):
                                kwargs['weights'] = torchvision.models.VGG16_Weights.IMAGENET1K_V1
                        return orig_vgg16(*args, **kwargs)
                    setattr(torchvision.models, 'vgg16', patched_vgg16)
                    setattr(torchvision.models, '_patched_vgg16_applied', True)
                
                import lpips
                # 2026: Mission Pulse - Restore transparency for slow perceptual engine loading
                print(" [MISSION] Initializing Neural Perceptual Engine (LPIPS/VGG16)...")
                # Natively trained perceptual alignment! Exponentially more stable than crude VGG L1
                
                class AutocastLPIPS(torch.nn.Module):
                    def __init__(self, perc_module):
                        super().__init__()
                        self.perc = perc_module
                    def forward(self, x, y):
                        with torch.amp.autocast('cuda', enabled=True):
                            return self.perc(x, y)
                            
                self.perc = AutocastLPIPS(lpips.LPIPS(net='vgg'))
                
                self.perc.eval()
                for param in self.perc.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"[WARNING] [RESILIENCE] LPIPS failed to bind ({e}). Defaulting to pure L1.")


    def forward(self, pred, target, task_idx=None):
        if self.task_type in ["restoration", "enhancement"]:
            # 2026: Branched Multi-Task Support (e.g., BranchedFFANet)
            if isinstance(target, dict) and isinstance(pred, dict):
                img_target = target["image"]
                img_pred = pred["restored_image"]
                base_loss = self.l1(img_pred, img_target)
                
                if self.perc is not None:
                    p_scaled = torch.clamp(img_pred, 0, 1) * 2.0 - 1.0
                    t_scaled = torch.clamp(img_target, 0, 1) * 2.0 - 1.0
                    raw_lpips = self.perc(p_scaled, t_scaled).view(-1)
                    if task_idx is not None and len(task_idx.shape) > 0:
                        dynamic_weights = self.task_lpips_weights[task_idx]
                        perc_loss = (raw_lpips * dynamic_weights).mean()
                    else:
                        perc_loss = 0.025 * raw_lpips.mean()
                    base_loss += perc_loss
                
                # Detection Loss (Best Practice: lambda = 0.1 to prevent detection gradients from destroying image quality)
                bbox_pred = pred["detections"]
                # In a full production implementation, this interfaces with Ultralytics anchor-matching.
                # Here we use a smooth_l1 placeholder against a dummy target of the same shape to ensure end-to-end graph flow.
                detect_loss = F.smooth_l1_loss(bbox_pred, torch.zeros_like(bbox_pred))
                
                return base_loss + 0.1 * detect_loss

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

            # NIMA specific Earth Mover's Distance (EMD) (Removed dangerous logit clamping that causes zero-gradients)
            p_probs = F.softmax(pred_f / float(self.stab.get("softmax_temp", 1.0)), dim=-1)
            t_probs = tgt_f / torch.clamp(tgt_f.sum(dim=-1, keepdim=True), min=float(self.stab.get("emd_epsilon", 1e-6)))

            cdf_p = torch.cumsum(p_probs, dim=-1)
            cdf_t = torch.cumsum(t_probs, dim=-1)

            # 2026: Geometric Stabilizer - Summing squared CDF error per-bin
            emd = torch.sum((cdf_p - cdf_t) ** 2, dim=-1).mean()

            # Expected score scalar calculation [B]
            p_mean = (p_probs * self.rank_weights).sum(dim=-1)
            t_mean = (t_probs * self.rank_weights).sum(dim=-1)

            # Cross-Microbatch Memory Buffer Injection
            if self.memory_bank is not None:
                p_eval, t_eval = self.memory_bank.get_context(p_mean, t_mean)
                self.memory_bank.update(p_mean, t_mean)
            else:
                p_eval, t_eval = p_mean, t_mean

            total_loss = emd

            # --- 2026: Differentiable Soft-Spearman Loss ---
            use_soft_spearman = self.stab.get("use_soft_spearman", True)
            if use_soft_spearman and p_eval.size(0) > 1:
                spearman_loss = self.soft_spearman(p_eval, t_eval)
                spearman_weight = float(self.stab.get("soft_spearman_weight", 0.5))
                total_loss = total_loss + (spearman_weight * spearman_loss)

            # --- 2026: Neural Rank-Boost (Pairwise Margin Loss) ---
            rank_weight = float(self.stab.get('rank_weight', 0.0))
            if rank_weight > 0 and p_eval.size(0) > 1:
                p_diff = p_eval.unsqueeze(0) - p_eval.unsqueeze(1)
                t_diff = t_eval.unsqueeze(0) - t_eval.unsqueeze(1)
                t_sign = torch.sign(t_diff)

                margin = float(self.stab.get('rank_margin', 0.05))
                rank_loss = F.relu(margin - t_sign * p_diff)
                mask = (t_sign != 0).float()
                avg_rank_loss = (rank_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                total_loss = total_loss + (rank_weight * avg_rank_loss)

            return total_loss
            
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
                
                with torch.amp.autocast('cuda', enabled=False):
                    loss_conf = F.binary_cross_entropy(p_conf.float(), t_conf.float())
                
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

