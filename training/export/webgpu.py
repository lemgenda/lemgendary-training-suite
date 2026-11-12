"""WebGPU-optimized ONNX model exporter with fixed shapes and slice-error prevention."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from training.export.base import EXPORT_ERRORS
from training.export.common import (
    DEFAULT_ONNX_OPSET,
    ExportError,
    OnnxGraphSpec,
    model_device,
    unwrap_export_model,
)
from training.export.dynamo_onnx import capture_program
from training.export.onnx_weights import WeightLayout, save_program

logger = logging.getLogger("lemtrain.export.webgpu")


def export_webgpu_onnx(
    model: nn.Module,
    save_path: Path | str,
    dummy_input_shape: tuple[int, ...] = (1, 3, 512, 512),
    opset: int = DEFAULT_ONNX_OPSET,
) -> bool:
    """Export a WebGPU payload with a fixed input shape and embedded weights (2 GB rule)."""
    out_file = Path(save_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting WebGPU ONNX payload to %s (opset=%d, shape=%s)...",
        out_file,
        opset,
        dummy_input_shape,
    )

    model_to_export = unwrap_export_model(model)
    model_to_export.eval()
    dummy_input = torch.randn(dummy_input_shape, device=model_device(model_to_export))

    try:
        program = capture_program(model_to_export, (dummy_input,), OnnxGraphSpec(opset=opset))
        save_program(program, out_file, WeightLayout.EMBEDDED)
        logger.info("WebGPU ONNX export successful! Opset: %d, Shape: %s", opset, dummy_input_shape)
        return True
    except EXPORT_ERRORS as exc:
        err_msg = f"WebGPU export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("WEBGPU_EXPORT_FAILED", err_msg) from exc
