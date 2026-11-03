"""Forex model exporter for LemGendary AI Training Suite.

Handles MetaTrader 5 Expert Advisor signal ONNX and PyTorch checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import external_data_helper
from onnxconverter_common import float16
from torch import nn

from training.export.base import (
    EXPORT_ERRORS,
    MODEL_IO_ERRORS,
    BaseModelExporter,
    spec_dataset_types,
)
from training.export.mt5_signal import export_forex_onnx, export_forex_onnx_from_model
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.forex")


def _save_forex_state_dict(model: Any, checkpoint_path: Path | str | None, pt_path: Path) -> bool:
    """Save the FP32 state dict from the model, else from the checkpoint; False if neither."""
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), str(pt_path))
        logger.info("Saved Forex FP32 PyTorch model state dict to %s", pt_path)
        return True
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        torch.save(state, str(pt_path))
        logger.info("Saved Forex FP32 PyTorch model state dict from checkpoint to %s", pt_path)
        return True
    return False


def _trace_forex_onnx(
    model: Any,
    checkpoint_path: Path | str | None,
    fp32_onnx_path: Path,
    active_timeframes: list[int] | None,
) -> Path | None:
    """Trace the model (or the checkpoint) to an FP32 ONNX graph; ``None`` if neither."""
    if isinstance(model, nn.Module):
        return export_forex_onnx_from_model(
            model=model,
            out_path=fp32_onnx_path,
            active_timeframes=active_timeframes,
        )
    if checkpoint_path and Path(checkpoint_path).exists():
        return export_forex_onnx(
            checkpoint_path=checkpoint_path,
            out_path=fp32_onnx_path,
            active_timeframes=active_timeframes,
        )
    logger.warning("No model instance or checkpoint file provided for Forex ONNX tracing.")
    return None


def _sidecar_and_fp16(
    fp32_onnx_path: Path, fp16_onnx_path: Path, exported_files: dict[str, Path]
) -> None:
    """Move FP32 weights to a ``.onnx.data`` sidecar, then write an integrated FP16 graph."""
    fp32_data_filename = f"{fp32_onnx_path.name}.data"
    # Load the full FP32 model into memory before converting to sidecar
    m_fp32 = onnx.load(str(fp32_onnx_path))

    # Convert FP32 model to external data sidecar (.onnx.data)
    external_data_helper.convert_model_to_external_data(
        m_fp32,
        all_tensors_to_one_file=True,
        location=fp32_data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(m_fp32, str(fp32_onnx_path))
    exported_files["onnx_fp32"] = fp32_onnx_path
    exported_files["onnx_fp32_data"] = fp32_onnx_path.parent / fp32_data_filename
    logger.info(
        "Exported Forex FP32 ONNX graph (%s) with binary data sidecar (%s)",
        fp32_onnx_path.name,
        fp32_data_filename,
    )

    # Load the model with external data to convert weights to float16
    m_for_fp16 = onnx.load(str(fp32_onnx_path), load_external_data=True)
    m_fp16 = float16.convert_float_to_float16(m_for_fp16)
    onnx.save(m_fp16, str(fp16_onnx_path))
    exported_files["onnx"] = fp16_onnx_path
    exported_files["onnx_fp16"] = fp16_onnx_path
    logger.info(
        "Exported Forex FP16 integrated ONNX model (%s)",
        fp16_onnx_path.name,
    )


class ForexModelExporter(BaseModelExporter):
    """Exports multi-timeframe ForexPredictor models for MT5 integration."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if model is ForexPredictor."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            d_types = spec_dataset_types(resolved_spec)
            return "forex" in d_types or "forex" in resolved_spec.key.lower()
        return "forex" in model_key.lower()

    def export(
        self,
        model: Any,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Export a Forex model to ONNX (FP32 sidecar + FP16 integrated) and PyTorch formats."""
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)

        checkpoint_path: Path | str | None = kwargs.get("checkpoint_path")
        active_timeframes: list[int] | None = kwargs.get("active_timeframes")
        if active_timeframes is None and resolved_spec:
            active_timeframes = (
                (resolved_spec.raw_config or {})
                .get("kwargs", {})
                .get("active_timeframes", [1, 5, 15, 60, 240, 1440])
            )

        exported_files: dict[str, Path] = {}

        # 1. Standalone FP32 PyTorch state dict (.pt)
        pt_path = target_dir / f"{model_key}.pt"
        try:
            if _save_forex_state_dict(model, checkpoint_path, pt_path):
                exported_files["pt"] = pt_path
        except MODEL_IO_ERRORS as exc:
            logger.warning("Failed saving Forex .pt state dict: %s", exc)

        # 2. FP32 ONNX Model with Separate Weights Sidecar (.onnx + .onnx.data)
        # 3. FP16 ONNX Model with Integrated Weights (.onnx)
        fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
        try:
            exported_base = _trace_forex_onnx(
                model, checkpoint_path, fp32_onnx_path, active_timeframes
            )
            if exported_base and exported_base.exists():
                _sidecar_and_fp16(fp32_onnx_path, target_dir / f"{model_key}.onnx", exported_files)
        except (*EXPORT_ERRORS, *MODEL_IO_ERRORS) as exc:
            logger.error("Failed Forex ONNX tri-format export: %s", exc)

        return exported_files
