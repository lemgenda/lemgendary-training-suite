import os
import time
import numpy as np
import torch

# ==============================================================================
# [LAUNCH] LemGendary 2026 Telemetry Engine (v18.0)
# ==============================================================================
# Master handler for SOTA metric tracking, Quality Score math, and CSV validation
# ==============================================================================

# --- 2026 Resilience: Extended Mathematical Governance ---
METRIC_WEIGHTS = {
    'plcc': 50, 'srcc': 50, 'psnr': 10, 'ssim': 40,
    'lpips': 40, 'fid': 1, 'map50': 100, 'map50_95': 100,
    'rank_margin': 20, 'accuracy': 100, 'mae': 100,
    # [2026: SOTA Expansion Targets]
    'miou': 100, 'map_medium': 100, 'map_hard': 100, 'accuracy_vqa': 100,
    'dir_acc': 1.0, 'win_rate': 1.0, 'tp_mae': 1.0, 'sl_mae': 1.0
}

METRIC_DIRECTIONS = {
    'plcc': True, 'srcc': True, 'psnr': True, 'ssim': True,
    'lpips': False, 'fid': False, 'map50': True, 'map50_95': True,
    'rank_margin': False, 'accuracy': True, 'mae': False,
    # [2026: SOTA Expansion Targets]
    'miou': True, 'map_medium': True, 'map_hard': True, 'accuracy_vqa': True,
    'dir_acc': True, 'win_rate': True, 'tp_mae': False, 'sl_mae': False
}

class TelemetryEngine:
    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        self.metrics_csv_path = os.path.join(export_dir, "metrics.csv")

    def validate_and_initialize_csv(self):
        """2026 Schema Guard: Force-rebuild or transition the CSV to 28-column hardware-aware parity."""
        schema_ok = False
        if os.path.exists(self.metrics_csv_path):
            try:
                with open(self.metrics_csv_path, "r", encoding='utf-8') as f:
                    header = f.readline().strip()
                    # 2026 Schema: 28 columns (v10.0 includes SOTA telemetry expansion)
                    if len(header.split(",")) == 28:
                        schema_ok = True
            except: pass

        if not schema_ok:
            legacy_path = self.metrics_csv_path.replace(".csv", "_legacy.csv")
            if os.path.exists(self.metrics_csv_path):
                try:
                    # Use atomic-friendly naming if legacy already exists
                    if os.path.exists(legacy_path):
                        legacy_path = legacy_path.replace(".csv", f"_{int(time.time())}.csv")
                    os.rename(self.metrics_csv_path, legacy_path)
                    print(f" [TELEMETRY] Legacy or corrupted metrics detected. Archiving to {os.path.basename(legacy_path)} and initializing 28-column SOTA log.")
                except: pass

            with open(self.metrics_csv_path, "w", encoding='utf-8') as f:
                f.write("Epoch,Train_Loss,Val_Loss,LR,PLCC,SRCC,PSNR,SSIM,LPIPS,FID,mAP50,mAP50-95,Accuracy,Rank_Margin,MAE,mIoU,mAP_Medium,mAP_Hard,Accuracy_VQA,Quality_Score,Res,Data,Temp,Clamp,Cooldown,Batch,Accumulation,Stress\n")

    def write_epoch_row(self, epoch, train_loss, val_loss, lr, curr_metrics, quality_score, governor_state, stress):
        """Mathematically serialize the 28-column row directly to the IO stream."""
        plcc = curr_metrics.get('plcc', 0.0)
        srcc = curr_metrics.get('srcc', 0.0)
        psnr = curr_metrics.get('psnr', 0.0)
        ssim = curr_metrics.get('ssim', 0.0)
        lpips = curr_metrics.get('lpips', 0.0)
        fid = curr_metrics.get('fid', 0.0)
        map50 = curr_metrics.get('map50', 0.0)
        map50_95 = curr_metrics.get('map50_95', 0.0)
        accuracy = curr_metrics.get('accuracy', 0.0)
        rank_margin = curr_metrics.get('rank_margin', 0.0)
        mae = curr_metrics.get('mae', 0.0)
        miou = curr_metrics.get('miou', 0.0)
        map_medium = curr_metrics.get('map_medium', 0.0)
        map_hard = curr_metrics.get('map_hard', 0.0)
        acc_vqa = curr_metrics.get('accuracy_vqa', 0.0)

        # 2026 Governor extraction variables
        res = governor_state.get('input_size', 512) if governor_state else 512
        data = governor_state.get('sample_fraction', 0.15) if governor_state else 0.15
        temp = governor_state.get('softmax_temp', 1.5) if governor_state else 1.5
        clamp = governor_state.get('logit_clamp', 15.0) if governor_state else 15.0
        cooldown = governor_state.get('cooldown_remaining', 0) if governor_state else 0
        batch = governor_state.get('batch_size', 1) if governor_state else 1
        accum = governor_state.get('accumulation_steps', 24) if governor_state else 24

        with open(self.metrics_csv_path, "a", encoding='utf-8') as f:
            f.write(f"{epoch+1},{train_loss:.8f},{val_loss:.8f},{lr:.8f},"
                    f"{plcc:.4f},{srcc:.4f},{psnr:.4f},{ssim:.4f},{lpips:.4f},{fid:.4f},"
                    f"{map50:.4f},{map50_95:.4f},{accuracy:.4f},{rank_margin:.4f},{mae:.4f},"
                    f"{miou:.4f},{map_medium:.4f},{map_hard:.4f},{acc_vqa:.4f},{quality_score:.4f},"
                    f"{res},{data:.2f},{temp:.4f},{clamp:.1f},{cooldown},{batch},{accum},{stress:.6f}\n")

    def compute_quality_score(self, curr_metrics, sota_targets, task_type):
        """Dynamic Quality Score: Weighted average of all SOTA targets."""
        quality_score = 0.0
        
        # --- 2026: Metric Singularity Shield & Live Polarity Shield (v1.2.2) ---
        if task_type == "quality":
            plcc_val = curr_metrics.get('plcc', 0.0)
            srcc_val = curr_metrics.get('srcc', 0.0)
            if np.isnan(plcc_val) or np.isnan(srcc_val) or plcc_val < 0.0 or srcc_val < 0.0:
                return 0.0, True # Return score, and a flag indicating singularity or polarity collapse
            
        for k, target_v in sota_targets.items():
            val = curr_metrics.get(k, 0.0)
            direction = METRIC_DIRECTIONS.get(k, True)
            weight = METRIC_WEIGHTS.get(k, 1)

            if direction:
                quality_score += val * weight
            else:
                # Inverted: We use standard 2026 normalization for restoration metrics
                if k == 'fid': quality_score += (100.0 - val) * weight
                elif k == 'lpips': quality_score += (1.0 - val) * weight
                elif k == 'rank_margin': quality_score += (10.0 - val) * weight # Margin is 0-9 scale
                else: quality_score += (1.0 / (val + 1e-6)) * weight
                
        return quality_score, False

    def calculate_miou(self, preds, targets):
        """2026 mIoU Integration for ParseNet."""
        # preds: logits [B, C, H, W]
        # targets: [B, H, W]
        try:
            pred_class = torch.argmax(preds, dim=1)
            intersection = torch.logical_and(targets > 0, pred_class == targets).sum().item()
            union = torch.logical_or(targets > 0, pred_class > 0).sum().item()
            if union == 0:
                return 0.0
            return float(intersection) / float(union)
        except Exception:
            return 0.0
            
    def calculate_vqa_accuracy(self, preds, targets):
        """2026 VQA Accuracy via string exact match or contained match."""
        try:
            if not isinstance(preds, list) or not isinstance(targets, list):
                return 0.0
            correct = 0
            for p, t in zip(preds, targets):
                p_clean = str(p).lower().strip()
                t_clean = str(t).lower().strip()
                if t_clean in p_clean or p_clean in t_clean:
                    correct += 1
            return float(correct) / max(1, len(targets))
        except Exception:
            return 0.0
            
    # Note: mAP_Medium and mAP_Hard should be calculated via PyCocoTools
    # This requires intercepting bounding boxes. For now we provide a skeleton.
    def calculate_map(self, preds, targets):
        """2026 mAP Hooks for Detection (YOLO/Face)."""
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            # Requires preds/targets to be lists of dicts: {'boxes': [N,4], 'scores': [N], 'labels': [N]}
            map_metric = MeanAveragePrecision(iou_type="bbox")
            map_metric.update(preds, targets)
            res = map_metric.compute()
            return res.get('map_50', torch.tensor(0.0)).item(), res.get('map', torch.tensor(0.0)).item()
        except Exception as e:
            print(f" [WARNING] mAP Eval failed: {e}")
            return 0.0, 0.0
