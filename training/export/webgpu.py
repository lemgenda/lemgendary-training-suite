"""WebGPU-optimized ONNX model exporter with fixed shapes and slice-error prevention."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn

from training.export.common import ExportError

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

    logger.info("Exporting WebGPU ONNX payload to %s (opset=%d, shape=%s)...", out_file, opset, dummy_input_shape)

    model_to_export: nn.Module
    if hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model_to_export = getattr(model, "module")
    elif isinstance(model, nn.Module):
        model_to_export = model
    else:
        model_to_export = model

    device = torch.device("cpu")
    if isinstance(model_to_export, nn.Module):
        model_to_export.eval()
        try:
            device = next(model_to_export.parameters()).device
        except StopIteration:
            pass

    dummy_input = torch.randn(dummy_input_shape, device=device)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            if not isinstance(model_to_export, nn.Module):
                return False
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                str(out_file),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,
                dynamo=False,
            )
        logger.info("WebGPU ONNX export successful! Opset: %d, Shape: %s", opset, dummy_input_shape)
        return True
    except Exception as exc:
        err_msg = f"WebGPU export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("WEBGPU_EXPORT_FAILED", err_msg) from exc
