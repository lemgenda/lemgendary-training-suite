"""Optimization engine legacy re-export module.

Maintains backward compatibility with pre-refactor pipelines by delegating
autonomous governance to training.governance.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from typing import Any

import torch

from training.governance.curriculum import CurriculumState
from training.governance.governor import GovernorStateError, SmartTrainingGovernor
from training.governance.metrics import (
    DEFAULT_METRIC_DIRECTIONS,
    DEFAULT_METRIC_WEIGHTS,
    MetricDefinition,
    MetricRegistry,
)
from training.governance.sota import SotaTracker
from training.export.webgpu import export_webgpu_onnx
from training.governance.thermal import ThermalState

logger = logging.getLogger("lemtrain.optimization_engine")


__all__ = [
    "CurriculumState",
    "DEFAULT_METRIC_DIRECTIONS",
    "DEFAULT_METRIC_WEIGHTS",
    "GovernorStateError",
    "MetricDefinition",
    "MetricRegistry",
    "SmartTrainingGovernor",
    "SotaTracker",
    "ThermalState",
    "export_webgpu_onnx",
]