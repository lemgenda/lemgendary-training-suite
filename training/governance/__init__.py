"""Governance package managing curriculum, thermal, metric registry, and SOTA tracking."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from training.governance.curriculum import CurriculumState
from training.governance.governor import GovernorStateError, SmartTrainingGovernor
from training.governance.metrics import (
    DEFAULT_METRIC_DIRECTIONS,
    DEFAULT_METRIC_WEIGHTS,
    MetricDefinition,
    MetricRegistry,
)
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState

if TYPE_CHECKING:
    from training.governance.yolo_governor import (
        YOLOCurriculumGovernor,
        run_governed_yolo_training,
    )

__all__ = [
    "DEFAULT_METRIC_DIRECTIONS",
    "DEFAULT_METRIC_WEIGHTS",
    "CurriculumState",
    "GovernorStateError",
    "MetricDefinition",
    "MetricRegistry",
    "SmartTrainingGovernor",
    "SotaTracker",
    "ThermalState",
    "YOLOCurriculumGovernor",
    "run_governed_yolo_training",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve YOLOCurriculumGovernor and run_governed_yolo_training."""
    if name in ("YOLOCurriculumGovernor", "run_governed_yolo_training"):
        yolo_governor = importlib.import_module("training.governance.yolo_governor")
        return getattr(yolo_governor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
