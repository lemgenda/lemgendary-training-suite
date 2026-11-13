"""Single-file ONNX export of an image model in one of the release variants."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from training.export.base import EXPORT_ERRORS
from training.export.common import (
    ExportError,
    OnnxGraphSpec,
    default_graph_spec,
    model_device,
    unwrap_export_model,
)
from training.export.onnx_variants import ONNX_FP16_VARIANT, ONNX_FP32_VARIANT, OnnxVariantWriter

logger = logging.getLogger("lemtrain.export.onnx")


def export_onnx(
    model: nn.Module,
    save_path: Path | str,
    dummy_shape: tuple[int, ...] = (2, 3, 256, 256),
    *,
    half: bool = False,
    graph: OnnxGraphSpec | None = None,
) -> bool:
    """Export ``model`` to ``save_path`` as the FP16 (``half``) or FP32 release variant.

    FP32 always writes the ``<file>.data`` weights sidecar; FP16 embeds its weights unless
    they exceed the protobuf limit. ``graph`` defaults to one ``input`` and one ``output``
    tensor with dynamic batch, height and width. The sample batch should be at least 2 so
    ``torch.export`` does not specialise the batch axis to 1.
    """
    out_file = Path(save_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_to_export = unwrap_export_model(model)
    model_to_export.eval()
    sample = torch.randn(dummy_shape, device=model_device(model_to_export))
    graph = graph or default_graph_spec(sample)
    variant = ONNX_FP16_VARIANT if half else ONNX_FP32_VARIANT

    logger.info(
        "Exporting %s ONNX model to %s (opset=%d, shape=%s)...",
        variant.key,
        out_file,
        graph.opset,
        dummy_shape,
    )
    try:
        OnnxVariantWriter(model_to_export, (sample,), graph).write(variant, out_file)
    except EXPORT_ERRORS as exc:
        err_msg = f"ONNX export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("ONNX_EXPORT_FAILED", err_msg) from exc
    logger.info("Successfully exported ONNX model to %s", out_file)
    return True
