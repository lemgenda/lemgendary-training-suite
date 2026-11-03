"""ONNX model export supporting dynamic axes, FP32, and FP16 half-precision."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from training.export.common import (
    ExportError,
    OnnxGraphSpec,
    detach_onnx_weights,
    model_device,
    run_onnx_export,
    unwrap_export_model,
)

logger = logging.getLogger("lemtrain.export.onnx")


def export_onnx(
    model: nn.Module,
    save_path: Path | str,
    dummy_shape: tuple[int, ...] = (1, 3, 256, 256),
    *,
    half: bool = False,
    graph: OnnxGraphSpec | None = None,
) -> bool:
    """Export PyTorch model to ONNX format.

    ``graph`` defaults to one ``input`` and one ``output`` tensor, static axes and opset 17.
    """
    out_file = Path(save_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    graph = graph or OnnxGraphSpec()

    model_to_export = unwrap_export_model(model)
    model_to_export.eval()
    device = model_device(model_to_export)
    target_dtype = torch.float16 if half else torch.float32
    if half:
        model_to_export = model_to_export.to(dtype=target_dtype)

    dummy_input = torch.randn(dummy_shape, dtype=target_dtype, device=device)

    logger.info(
        "Exporting ONNX model to %s (opset=%d, half=%s, shape=%s)...",
        out_file,
        graph.opset,
        half,
        dummy_shape,
    )

    try:
        run_onnx_export(model_to_export, (dummy_input,), out_file, graph)

        if graph.external_data:
            data_filename = f"{out_file.name}.data"
            detach_onnx_weights(out_file, data_filename)
            logger.info("Converted FP32 ONNX to external weights sidecar: %s", data_filename)

        logger.info("Successfully exported ONNX model to %s", out_file)
        return True
    except Exception as exc:
        err_msg = f"ONNX export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("ONNX_EXPORT_FAILED", err_msg) from exc

    finally:
        if half:
            # Revert model to float32
            model_to_export.to(dtype=torch.float32)
