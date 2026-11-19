"""Canonical schemas and definitions for metrics.csv and history.csv recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskFamily(str, Enum):
    """Supported model/task families in the LemGendary ecosystem."""

    VISION = "vision"
    FOREX = "forex"
    YOLO = "yolo"
    FACE_DETECTION = "face_detection"
    LLM = "llm"


# -----------------------------------------------------------------------------
# Canonical Column Definitions
# -----------------------------------------------------------------------------

# Common Core Step Columns
COMMON_STEP_COLUMNS: tuple[str, ...] = (
    "step",
    "epoch",
    "timestamp",
    "loss",
    "lr",
    "grad_norm",
    "throughput",
    "step_time",
    "vram_gb",
    "is_spike",
    "is_non_finite",
    "governor_action",
)

# Canonical Columns for History (Epoch / Evaluation Level)
COMMON_HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "timestamp",
    "train_loss",
    "val_loss",
    "lr",
    "grad_norm",
    "throughput",
    "memory_allocated_gb",
    "memory_peak_gb",
    "wall_time_sec",
    "governor_action",
    "is_spike",
    "is_non_finite",
    "is_best",
    "quality_score",
    "checkpoint_path",
)

# Family-specific metrics for History
VISION_METRIC_COLUMNS: tuple[str, ...] = (
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
    "map_medium",
    "map_hard",
    "accuracy_vqa",
)

FOREX_METRIC_COLUMNS: tuple[str, ...] = (
    "phase",
    "fold",
    "pairs",
    "dir_acc",
    "win_rate",
    "profit_factor",
    "sharpe",
    "sortino",
    "max_drawdown",
    "tp_mae",
    "sl_mae",
    "dir_entropy",
)

YOLO_METRIC_COLUMNS: tuple[str, ...] = (
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "map50",
    "map50_95",
    "top_class_name",
    "top_class_ap50",
    "top_class_ap50_95",
)

FACE_DETECTION_METRIC_COLUMNS: tuple[str, ...] = (
    "box_iou",
    "ap50",
    "ap50_95",
    "landmark_nme",
)

LLM_METRIC_COLUMNS: tuple[str, ...] = (
    "eval_loss",
    "perplexity",
    "token_throughput",
)

GOVERNOR_TELEMETRY_COLUMNS: tuple[str, ...] = (
    "res",
    "data_fraction",
    "softmax_temp",
    "logit_clamp",
    "cooldown_remaining",
    "batch_size",
    "accumulation_steps",
    "stress",
)


@dataclass(frozen=True)
class StepRecord:
    """Telemetry data captured at each optimizer/training step."""

    step: int
    epoch: int
    timestamp: float
    loss: float
    lr: float
    grad_norm: float
    throughput: float
    step_time: float
    vram_gb: float
    is_spike: bool = False
    is_non_finite: bool = False
    governor_action: str = "none"
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step record to dictionary suitable for CSV serialization."""
        data: dict[str, Any] = {
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": f"{self.timestamp:.4f}",
            "loss": f"{self.loss:.6f}",
            "lr": f"{self.lr:.8f}",
            "grad_norm": f"{self.grad_norm:.6f}",
            "throughput": f"{self.throughput:.2f}",
            "step_time": f"{self.step_time:.4f}",
            "vram_gb": f"{self.vram_gb:.3f}",
            "is_spike": int(self.is_spike),
            "is_non_finite": int(self.is_non_finite),
            "governor_action": self.governor_action,
        }
        for k, v in self.extra_metrics.items():
            if isinstance(v, float):
                data[k] = f"{v:.6f}"
            else:
                data[k] = str(v)
        return data


@dataclass(frozen=True)
class HistoryRecord:
    """Telemetry data captured at each epoch or evaluation pass."""

    epoch: int
    timestamp: float
    train_loss: float
    val_loss: float
    lr: float
    grad_norm: float
    throughput: float
    memory_allocated_gb: float
    memory_peak_gb: float
    wall_time_sec: float
    governor_action: str = "none"
    is_spike: bool = False
    is_non_finite: bool = False
    is_best: bool = False
    quality_score: float = 0.0
    checkpoint_path: str = ""
    family: TaskFamily = TaskFamily.VISION
    family_metrics: dict[str, Any] = field(default_factory=dict)
    governor_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert history record to a flat dictionary for CSV serialization."""
        data: dict[str, Any] = {
            "epoch": self.epoch,
            "timestamp": f"{self.timestamp:.4f}",
            "train_loss": f"{self.train_loss:.6f}",
            "val_loss": f"{self.val_loss:.6f}",
            "lr": f"{self.lr:.8f}",
            "grad_norm": f"{self.grad_norm:.6f}",
            "throughput": f"{self.throughput:.2f}",
            "memory_allocated_gb": f"{self.memory_allocated_gb:.3f}",
            "memory_peak_gb": f"{self.memory_peak_gb:.3f}",
            "wall_time_sec": f"{self.wall_time_sec:.2f}",
            "governor_action": self.governor_action,
            "is_spike": int(self.is_spike),
            "is_non_finite": int(self.is_non_finite),
            "is_best": int(self.is_best),
            "quality_score": f"{self.quality_score:.4f}",
            "checkpoint_path": self.checkpoint_path,
        }

        # Inject family metrics
        for k, v in self.family_metrics.items():
            if v is None:
                data[k] = ""
            elif isinstance(v, float):
                data[k] = f"{v:.4f}"
            else:
                data[k] = str(v)

        # Inject governor telemetry
        for k, v in self.governor_state.items():
            if v is None:
                data[k] = ""
            elif isinstance(v, float):
                data[k] = f"{v:.4f}"
            else:
                data[k] = str(v)

        return data
