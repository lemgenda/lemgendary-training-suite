"""ONNX model export supporting dynamic axes, FP32, and FP16 half-precision."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.export.common import ExportError

logger = logging.getLogger("lemtrain.export.onnx")


def export_onnx(
    model: nn.Module,
    save_path: Path | str,
    dummy_shape: tuple[int, ...] = (1, 3, 256, 256),
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset: int = 17,
    half: bool = False,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> bool:
    """Export PyTorch model to ONNX format."""
    out_file = Path(save_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    in_names = input_names or ["input"]
    out_names = output_names or ["output"]

    model_to_export = model.module if hasattr(model, "module") else model
    model_to_export.eval()

    device = torch.device("cpu")
    try:
        device = next(model_to_export.parameters()).device
    except StopIteration:
        pass

    target_dtype = torch.float16 if half else torch.float32
    if half:
        model_to_export = model_to_export.to(dtype=target_dtype)

    dummy_input = torch.randn(dummy_shape, dtype=target_dtype, device=device)

    logger.info(
        "Exporting ONNX model to %s (opset=%d, half=%s, shape=%s)...",
        out_file,
        opset,
        half,
        dummy_shape,
    )

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            torch.onnx.export(
                model_to_export,
                (dummy_input,),
                str(out_file),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dynamic_axes,
            )
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
