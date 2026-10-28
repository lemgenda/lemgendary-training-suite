import os
import time
import numpy as np
import torch

# ==============================================================================
# [LAUNCH] LemGendary 2026 Telemetry Engine (v18.0)
# ==============================================================================
# Master handler for SOTA metric tracking, Quality Score math, and CSV validation
# ==============================================================================

from training.governance.metrics import (
    DEFAULT_METRIC_DIRECTIONS,
    DEFAULT_METRIC_WEIGHTS,
)

# --- 2026 Resilience: Extended Mathematical Governance (SSOT Delegation) ---
METRIC_WEIGHTS = DEFAULT_METRIC_WEIGHTS
METRIC_DIRECTIONS = DEFAULT_METRIC_DIRECTIONS

class TelemetryEngine:
    def __init__(self, export_dir: str, task_type: str = "image"):
        self.export_dir = export_dir
        self.task_type = task_type
        self.metrics_csv_path = os.path.join(export_dir, "metrics.csv")

    def validate_and_initialize_csv(self):
        """2026 Schema Guard: Force-rebuild or transition the CSV to hardware-aware parity."""
        schema_ok = False
        
        # Determine expected columns based on task_type
        if self.task_type == "forex":
            expected_cols = 24
            header_str = "Phase,Fold,Epoch,Train_Loss,Val_Loss,LR,DirAcc,WinRate,ProfitFactor,Sharpe,Sortino,MaxDD,TP_MAE,SL_MAE,DirEntropy,Quality_Score,Pairs,Data,Temp,Clamp,Cooldown,Batch,Accumulation,Stress\n"
        else:
            expected_cols = 28
            header_str = "Epoch,Train_Loss,Val_Loss,LR,PLCC,SRCC,PSNR,SSIM,LPIPS,FID,mAP50,mAP50-95,Accuracy,Rank_Margin,MAE,mIoU,mAP_Medium,mAP_Hard,Accuracy_VQA,Quality_Score,Res,Data,Temp,Clamp,Cooldown,Batch,Accumulation,Stress\n"

        if os.path.exists(self.metrics_csv_path):
            try:
                with open(self.metrics_csv_path, "r", encoding='utf-8') as f:
                    header = f.readline().strip()
                    if len(header.split(",")) == expected_cols:
                        schema_ok = True
            except Exception as e:
                print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

        if not schema_ok:
            legacy_path = self.metrics_csv_path.replace(".csv", "_legacy.csv")
            if os.path.exists(self.metrics_csv_path):
                try:
                    # Use atomic-friendly naming if legacy already exists
                    if os.path.exists(legacy_path):
                        legacy_path = legacy_path.replace(".csv", f"_{int(time.time())}.csv")
                    os.rename(self.metrics_csv_path, legacy_path)
                    print(f" [TELEMETRY] Legacy or corrupted metrics detected. Archiving to {os.path.basename(legacy_path)} and initializing {expected_cols}-column SOTA log.")
                except Exception as e:
                    print(f"[REMEDY] Exception suppressed in telemetry/optimization: {e}")

            with open(self.metrics_csv_path, "w", encoding='utf-8') as f:
                f.write(header_str)

    def write_epoch_row(self, epoch, train_loss, val_loss, lr, curr_metrics, quality_score, governor_state, stress, num_pairs=None, phase=1, fold=1):
        """Mathematically serialize the row directly to the IO stream with deduplication and schema safety."""
        self.validate_and_initialize_csv()

        # 2026 Governor extraction variables
        data = governor_state.get('sample_fraction', 0.15) if governor_state else 0.15
        temp = governor_state.get('softmax_temp', 1.5) if governor_state else 1.5
        clamp = governor_state.get('logit_clamp', 15.0) if governor_state else 15.0
        cooldown = governor_state.get('cooldown_remaining', 0) if governor_state else 0
        batch = governor_state.get('batch_size', 1) if governor_state else 1
        accum = governor_state.get('accumulation_steps', 24) if governor_state else 24

        epoch_idx = int(epoch) if int(epoch) >= 1 else int(epoch) + 1

        if self.task_type == "forex":
            dir_acc = float(curr_metrics.get('dir_acc', 0.0))
            win_rate = float(curr_metrics.get('win_rate', 0.0))
            profit_factor = float(curr_metrics.get('profit_factor', 0.0))
            sharpe = float(curr_metrics.get('sharpe_ratio', 0.0))
            sortino = float(curr_metrics.get('sortino_ratio', 0.0))
            max_dd = float(curr_metrics.get('max_drawdown', 0.0))
            tp_mae = float(curr_metrics.get('tp_mae', 0.0))
            sl_mae = float(curr_metrics.get('sl_mae', 0.0))
            raw_ent = curr_metrics.get('dir_entropy', 0.0)
            dir_entropy = 0.0 if (raw_ent is None or np.isinf(raw_ent) or np.isnan(raw_ent)) else float(raw_ent)
            pairs = num_pairs if num_pairs is not None else (governor_state.get('input_size', 16) if governor_state else 16)

            new_line = (
                f"{phase},{fold},{epoch_idx},{train_loss:.8f},{val_loss:.8f},{lr:.8f},"
                f"{dir_acc:.4f},{win_rate:.4f},{profit_factor:.4f},{sharpe:.4f},{sortino:.4f},"
                f"{max_dd:.4f},{tp_mae:.4f},{sl_mae:.4f},{dir_entropy:.4f},{quality_score:.4f},"
                f"{pairs},{data:.2f},{temp:.4f},{clamp:.1f},{cooldown},{batch},{accum},{stress:.6f}\n"
            )
            key_tuple = (str(phase), str(fold), str(epoch_idx))
        else:
            plcc = float(curr_metrics.get('plcc', 0.0))
            srcc = float(curr_metrics.get('srcc', 0.0))
            psnr = float(curr_metrics.get('psnr', 0.0))
            ssim = float(curr_metrics.get('ssim', 0.0))
            lpips = float(curr_metrics.get('lpips', 0.0))
            fid = float(curr_metrics.get('fid', 0.0))
            map50 = float(curr_metrics.get('map50', 0.0))
            map50_95 = float(curr_metrics.get('map50_95', 0.0))
            accuracy = float(curr_metrics.get('accuracy', 0.0))
            rank_margin = float(curr_metrics.get('rank_margin', 0.0))
            mae = float(curr_metrics.get('mae', 0.0))
            miou = float(curr_metrics.get('miou', 0.0))
            map_medium = float(curr_metrics.get('map_medium', 0.0))
            map_hard = float(curr_metrics.get('map_hard', 0.0))
            acc_vqa = float(curr_metrics.get('accuracy_vqa', 0.0))
            res = governor_state.get('input_size', 512) if governor_state else 512

            new_line = (
                f"{epoch_idx},{train_loss:.8f},{val_loss:.8f},{lr:.8f},"
                f"{plcc:.4f},{srcc:.4f},{psnr:.4f},{ssim:.4f},{lpips:.4f},{fid:.4f},"
                f"{map50:.4f},{map50_95:.4f},{accuracy:.4f},{rank_margin:.4f},{mae:.4f},"
                f"{miou:.4f},{map_medium:.4f},{map_hard:.4f},{acc_vqa:.4f},{quality_score:.4f},"
                f"{res},{data:.2f},{temp:.4f},{clamp:.1f},{cooldown},{batch},{accum},{stress:.6f}\n"
            )
            key_tuple = (str(epoch_idx),)

        # Read existing file to update in-place if epoch already present, or append
        existing_lines = []
        if os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "r", encoding="utf-8") as f:
                existing_lines = f.readlines()

        if not existing_lines:
            with open(self.metrics_csv_path, "w", encoding="utf-8") as f:
                f.write(new_line)
            return

        header = existing_lines[0]
        data_rows = existing_lines[1:]
        replaced = False
        updated_rows = []

        for row in data_rows:
            parts = [p.strip() for p in row.split(",")]
            if self.task_type == "forex" and len(parts) >= 3:
                row_key = (parts[0], parts[1], parts[2])
            elif len(parts) >= 1:
                row_key = (parts[0],)
            else:
                row_key = ()

            if row_key == key_tuple:
                updated_rows.append(new_line)
                replaced = True
            else:
                updated_rows.append(row if row.endswith("\n") else row + "\n")

        if not replaced:
            updated_rows.append(new_line)

        with open(self.metrics_csv_path, "w", encoding="utf-8") as f:
            f.write(header if header.endswith("\n") else header + "\n")
            f.writelines(updated_rows)

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
                if k == 'fid': quality_score += (100.0 - val) * weight
                elif k == 'lpips': quality_score += (1.0 - val) * weight
                elif k == 'rank_margin': quality_score += (10.0 - val) * weight
                elif k == 'max_drawdown': quality_score += max(0.0, 100.0 - val) * weight
                elif k in ['tp_mae', 'sl_mae']: quality_score += max(0.0, 50.0 - val) * weight
                # dir_entropy: lower is better. Max theoretical entropy for 3 classes = ln(3) ~= 1.099
                # Contribution is 0 when entropy = ln(3) (random), and rises as model becomes confident.
                elif k == 'dir_entropy': quality_score += max(0.0, 1.099 - val) * weight
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
