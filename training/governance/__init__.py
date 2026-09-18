"""Governance package managing curriculum, thermal, metric registry, and SOTA tracking."""

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

__all__ = [
    "CurriculumState",
    "GovernorStateError",
    "SmartTrainingGovernor",
    "MetricDefinition",
    "MetricRegistry",
    "DEFAULT_METRIC_DIRECTIONS",
    "DEFAULT_METRIC_WEIGHTS",
    "SotaTracker",
    "ThermalState",
]
