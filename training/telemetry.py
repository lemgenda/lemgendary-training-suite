"""Telemetry engine: metrics.csv schema management, row logging, and Quality Score math."""

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# ==============================================================================
# [LAUNCH] LemGendary 2026 Telemetry Engine (v18.0)
# ==============================================================================
# Master handler for SOTA metric tracking, Quality Score math, and CSV validation
# ==============================================================================
from training.governance.metrics import FACE_DETECTION_METRICS
from training.governance.quality_score import score_metrics
from training.utils.lazy_imports import load_mean_average_precision

# --- 2026 Resilience: Extended Mathematical Governance (SSOT Delegation) ---
logger = logging.getLogger("lemtrain.telemetry")


@dataclass(frozen=True)
class EpochLosses:
    """Train/validation loss and learning rate of one epoch."""

    train_loss: float
    val_loss: float
    lr: float


@dataclass(frozen=True)
class ForexRowTag:
    """Forex-only columns: pair count (``None`` uses the governor's ``input_size``), phase, fold."""

    num_pairs: int | None = None
    phase: int = 1
    fold: int = 1


@dataclass(frozen=True)
class EpochRow:
    """One metrics.csv row: losses, metrics, Quality Score and the pre-audit governor state.

    ``forex`` is written for Forex models only.
    """

    epoch: int
    losses: EpochLosses
    metrics: dict[str, Any]
    quality_score: float
    governor_state: dict[str, Any] | None
    stress: float
    forex: ForexRowTag = ForexRowTag()


def _governor_columns(gov: dict[str, Any] | None) -> str:
    """Format the Data,Temp,Clamp,Cooldown,Batch,Accumulation columns."""
    data = gov.get("sample_fraction", 0.15) if gov else 0.15
    temp = gov.get("softmax_temp", 1.5) if gov else 1.5
    clamp = gov.get("logit_clamp", 15.0) if gov else 15.0
    cooldown = gov.get("cooldown_remaining", 0) if gov else 0
    batch = gov.get("batch_size", 1) if gov else 1
    accum = gov.get("accumulation_steps", 24) if gov else 24
    return f"{data:.2f},{temp:.4f},{clamp:.1f},{cooldown},{batch},{accum}"


def _epoch_index(epoch: int) -> int:
    return int(epoch) if int(epoch) >= 1 else int(epoch) + 1


def _cell(value: Any) -> str:
    """A ``.4f`` metric cell; empty when the value was not measured (absent, NaN, None).

    An unmeasured metric is never written as ``0`` or as a substitute value (R2-T10);
    ``+inf``/``-inf`` (e.g. the profit factor of a run without losses) are kept.
    """
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(number) else f"{number:.4f}"


def _metric_columns(metrics: dict[str, Any], names: tuple[str, ...]) -> str:
    """Format ``names`` from ``metrics`` as comma-separated cells (missing = empty)."""
    return ",".join(_cell(metrics.get(name)) for name in names)


_FOREX_METRICS = (
    "dir_acc",
    "win_rate",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "tp_mae",
    "sl_mae",
)
_VISION_METRICS = (
    "plcc",
    "srcc",
    "psnr",
    "ssim",
    "lpips",
    "fid",
    "map50",
    "map50_95",
    "accuracy",
    "rank_margin",
    "mae",
    "miou",
    # No model measures these two any more; the columns stay so existing vision logs keep
    # their 28-column layout.
    "map_medium",
    "map_hard",
    "accuracy_vqa",
)


def _forex_line(row: EpochRow) -> tuple[str, tuple[str, ...]]:
    """Return the 24-column Forex CSV line and its (phase, fold, epoch) key."""
    gov = row.governor_state
    epoch_idx = _epoch_index(row.epoch)
    pairs = (
        row.forex.num_pairs
        if row.forex.num_pairs is not None
        else (gov.get("input_size", 16) if gov else 16)
    )
    new_line = (
        f"{row.forex.phase},{row.forex.fold},{epoch_idx},{row.losses.train_loss:.8f},"
        f"{row.losses.val_loss:.8f},{row.losses.lr:.8f},"
        f"{_metric_columns(row.metrics, _FOREX_METRICS)},{_cell(row.metrics.get('dir_entropy'))},"
        f"{_cell(row.quality_score)},{pairs},{_governor_columns(gov)},{row.stress:.6f}\n"
    )
    return new_line, (str(row.forex.phase), str(row.forex.fold), str(epoch_idx))


def _spatial_line(row: EpochRow, names: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Return a spatial-model CSV line with the ``names`` metric columns and its (epoch,) key."""
    gov = row.governor_state
    epoch_idx = _epoch_index(row.epoch)
    res = gov.get("input_size", 512) if gov else 512
    new_line = (
        f"{epoch_idx},{row.losses.train_loss:.8f},{row.losses.val_loss:.8f},{row.losses.lr:.8f},"
        f"{_metric_columns(row.metrics, names)},{_cell(row.quality_score)},"
        f"{res},{_governor_columns(gov)},{row.stress:.6f}\n"
    )
    return new_line, (str(epoch_idx),)


def _vision_line(row: EpochRow) -> tuple[str, tuple[str, ...]]:
    """Return the 28-column vision CSV line and its (epoch,) key."""
    return _spatial_line(row, _VISION_METRICS)


def _face_detection_line(row: EpochRow) -> tuple[str, tuple[str, ...]]:
    """Return the 17-column single-box face detection CSV line and its (epoch,) key."""
    return _spatial_line(row, FACE_DETECTION_METRICS)


@dataclass(frozen=True)
class CsvLayout:
    """Header and row writer of one ``metrics.csv`` layout."""

    header: str
    line: Callable[[EpochRow], tuple[str, tuple[str, ...]]]

    @property
    def columns(self) -> int:
        """Number of columns of the layout."""
        return len(self.header.strip().split(","))


_SPATIAL_TAIL = "Quality_Score,Res,Data,Temp,Clamp,Cooldown,Batch,Accumulation,Stress\n"
FOREX_LAYOUT = CsvLayout(
    header=(
        "Phase,Fold,Epoch,Train_Loss,Val_Loss,LR,DirAcc,WinRate,ProfitFactor,"
        "Sharpe,Sortino,MaxDD,TP_MAE,SL_MAE,DirEntropy,Quality_Score,Pairs,Data,"
        "Temp,Clamp,Cooldown,Batch,Accumulation,Stress\n"
    ),
    line=_forex_line,
)
VISION_LAYOUT = CsvLayout(
    header=(
        "Epoch,Train_Loss,Val_Loss,LR,PLCC,SRCC,PSNR,SSIM,LPIPS,FID,mAP50,mAP50-95,"
        "Accuracy,Rank_Margin,MAE,mIoU,mAP_Medium,mAP_Hard,Accuracy_VQA," + _SPATIAL_TAIL
    ),
    line=_vision_line,
)
FACE_DETECTION_LAYOUT = CsvLayout(
    header=("Epoch,Train_Loss,Val_Loss,LR,Box_IoU,AP50,AP50_95,Landmark_NME," + _SPATIAL_TAIL),
    line=_face_detection_line,
)
"""RetinaFace: the measured single-box metrics (``training.training.face_detection_metrics``)."""

_LAYOUTS: dict[str, CsvLayout] = {
    "forex": FOREX_LAYOUT,
    "face_detection": FACE_DETECTION_LAYOUT,
}


def csv_layout(task_type: str) -> CsvLayout:
    """The ``metrics.csv`` layout of ``task_type`` (the vision layout by default)."""
    return _LAYOUTS.get(task_type, VISION_LAYOUT)


class TelemetryEngine:
    """Owns a model's metrics.csv and computes Quality Score and auxiliary task metrics."""

    def __init__(self, export_dir: str, task_type: str = "image"):
        self.export_dir = export_dir
        self.task_type = task_type
        self.metrics_csv_path = os.path.join(export_dir, "metrics.csv")

    def validate_and_initialize_csv(self):
        """2026 Schema Guard: Force-rebuild or transition the CSV to hardware-aware parity."""
        schema_ok = False
        layout = csv_layout(self.task_type)
        expected_cols = layout.columns
        header_str = layout.header

        if os.path.exists(self.metrics_csv_path):
            try:
                with open(self.metrics_csv_path, "r", encoding="utf-8") as f:
                    header = f.readline().strip()
                    if len(header.split(",")) == expected_cols:
                        schema_ok = True
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(
                    "Could not read metrics CSV header %s; treating schema as legacy: %s",
                    self.metrics_csv_path,
                    e,
                )

        if not schema_ok:
            legacy_path = self.metrics_csv_path.replace(".csv", "_legacy.csv")
            if os.path.exists(self.metrics_csv_path):
                try:
                    # Use atomic-friendly naming if legacy already exists
                    if os.path.exists(legacy_path):
                        legacy_path = legacy_path.replace(".csv", f"_{int(time.time())}.csv")
                    os.rename(self.metrics_csv_path, legacy_path)
                    print(
                        " [TELEMETRY] Legacy or corrupted metrics detected. Archiving to "
                        f"{os.path.basename(legacy_path)} and initializing "
                        f"{expected_cols}-column SOTA log."
                    )
                except OSError as e:
                    logger.warning(
                        "Could not archive legacy metrics CSV %s -> %s: %s",
                        self.metrics_csv_path,
                        legacy_path,
                        e,
                    )

            with open(self.metrics_csv_path, "w", encoding="utf-8") as f:
                f.write(header_str)

    def write_epoch_row(self, row: EpochRow) -> None:
        """Append one epoch row to metrics.csv with deduplication and schema safety."""
        self.validate_and_initialize_csv()
        new_line, key_tuple = csv_layout(self.task_type).line(row)
        self._upsert_line(new_line, key_tuple)

    def _upsert_line(self, new_line: str, key_tuple: tuple[str, ...]) -> None:
        """Replace the row whose key equals ``key_tuple``, else append ``new_line``."""
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
        replaced = False
        updated_rows = []

        for row in existing_lines[1:]:
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
        """Quality score of one epoch's metrics (see ``training.governance.quality_score``).

        Restoration, enhancement and face models score ``PSNR + 20*SSIM - 20*LPIPS`` on
        every epoch; FID stays out of the score. Other tasks use the weighted sum of the
        SOTA targets measured this epoch; an absent target contributes nothing.
        Returns ``(score, collapsed)``; ``collapsed`` flags a NaN or negative PLCC/SRCC
        of a quality model (Metric Singularity and Polarity Shield).
        """
        if task_type == "quality":
            plcc_val = curr_metrics.get("plcc", 0.0)
            srcc_val = curr_metrics.get("srcc", 0.0)
            if np.isnan(plcc_val) or np.isnan(srcc_val) or plcc_val < 0.0 or srcc_val < 0.0:
                return 0.0, True
        return score_metrics(curr_metrics, sota_targets, task_type), False

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
        except (RuntimeError, TypeError, AttributeError, IndexError) as exc:
            logger.warning("mIoU computation failed; reporting 0.0: %s", exc)
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
        except (TypeError, ValueError) as exc:
            logger.warning("VQA accuracy computation failed; reporting 0.0: %s", exc)
            return 0.0

    # Multi-box COCO mAP. The single-box RetinaFace metrics live in
    # ``training.training.face_detection_metrics``.
    def calculate_map(self, preds, targets):
        """2026 mAP Hooks for Detection (YOLO/Face)."""
        try:
            # preds/targets: lists of dicts {'boxes': [N,4], 'scores': [N], 'labels': [N]}
            map_metric = load_mean_average_precision()(iou_type="bbox")
            map_metric.update(preds, targets)
            res = map_metric.compute()
            return res.get("map_50", torch.tensor(0.0)).item(), res.get(
                "map", torch.tensor(0.0)
            ).item()
        except (ImportError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("mAP evaluation failed; reporting 0.0: %s", e)
            return 0.0, 0.0
