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
from training.governance.thermal import ThermalState

logger = logging.getLogger("lemtrain.optimization_engine")


def export_webgpu_onnx(
    model: torch.nn.Module,
    save_path: str,
    dummy_input_shape: tuple[int, ...] = (1, 3, 512, 512),
) -> bool:
    """Export zero-copy WebGPU sharing payload with fixed shape and Opset 17."""
    logger.info("Exporting zero-copy WebGPU sharing payload to %s...", save_path)

    model_to_export = model.module if hasattr(model, "module") else model
    model_to_export.eval()

    try:
        device = next(model_to_export.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    dummy_input = torch.randn(dummy_input_shape, device=device)

    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                save_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,
            )
        logger.info("WebGPU ONNX export successful! Opset: 17, Shape: %s", dummy_input_shape)
        return True
    except (RuntimeError, ValueError) as e:
        logger.error("WebGPU export failed: %s", e)
        return False


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