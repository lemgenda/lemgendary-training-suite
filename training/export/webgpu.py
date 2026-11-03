"""WebGPU-optimized ONNX model exporter with fixed shapes and slice-error prevention."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from training.export.common import (
    ExportError,
    OnnxGraphSpec,
    model_device,
    run_onnx_export,
    unwrap_export_model,
)

logger = logging.getLogger("lemtrain.export.webgpu")


def export_webgpu_onnx(
    model: nn.Module,
    save_path: Path | str,
    dummy_input_shape: tuple[int, ...] = (1, 3, 512, 512),
    opset: int = 17,
) -> bool:
    """Export zero-copy WebGPU sharing payload with fixed shape and Opset 17."""
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
        run_onnx_export(
            model_to_export,
            (dummy_input,),
            out_file,
            OnnxGraphSpec(opset=opset),
        )
        logger.info("WebGPU ONNX export successful! Opset: %d, Shape: %s", opset, dummy_input_shape)
        return True
    except Exception as exc:
        err_msg = f"WebGPU export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("WEBGPU_EXPORT_FAILED", err_msg) from exc
