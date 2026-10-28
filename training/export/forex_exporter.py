"""Forex model exporter for LemGendary AI Training Suite.

Handles MetaTrader 5 Expert Advisor signal ONNX and PyTorch checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import onnx
from onnxconverter_common import float16
import torch
import torch.nn as nn

from training.export.base import BaseModelExporter
from training.export.mt5_signal import export_forex_onnx, export_forex_onnx_from_model
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.forex")


class ForexModelExporter(BaseModelExporter):
    """Exports multi-timeframe ForexPredictor models for MT5 integration."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if model is ForexPredictor."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            d_type = resolved_spec.dataset_type
            d_types = [x.lower() for x in d_type] if isinstance(d_type, list) else [(d_type or "").lower()]
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
        """Export Forex model into ONNX (FP32 sidecar + FP16 integrated) and PyTorch checkpoint formats."""
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)

        checkpoint_path: Path | str | None = kwargs.get("checkpoint_path")
        active_timeframes: list[int] | None = kwargs.get("active_timeframes")
        if active_timeframes is None and resolved_spec:
            raw_cfg = resolved_spec.raw_config or {}
            active_timeframes = raw_cfg.get("kwargs", {}).get("active_timeframes", [1, 5, 15, 60, 240, 1440])

        exported_files: dict[str, Path] = {}

        # ---------------------------------------------------------------------
        # 1. Standalone FP32 PyTorch state dict (.pt)
        # ---------------------------------------------------------------------
        pt_path = target_dir / f"{model_key}.pt"
        try:
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), str(pt_path))
                exported_files["pt"] = pt_path
                logger.info("Saved Forex FP32 PyTorch model state dict to %s", pt_path)
            elif checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
                state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
                torch.save(state, str(pt_path))
                exported_files["pt"] = pt_path
                logger.info("Saved Forex FP32 PyTorch model state dict from checkpoint to %s", pt_path)
        except Exception as exc:
            logger.warning("Failed saving Forex .pt state dict: %s", exc)

        # ---------------------------------------------------------------------
        # 2. FP32 ONNX Model with Separate Weights Sidecar (.onnx + .onnx.data)
        # ---------------------------------------------------------------------
        fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
        fp32_data_filename = f"{model_key}_FP32.onnx.data"
        fp32_data_path = target_dir / fp32_data_filename

        try:
            exported_base: Path | None = None
            if isinstance(model, nn.Module):
                exported_base = export_forex_onnx_from_model(
                    model=model,
                    out_path=fp32_onnx_path,
                    active_timeframes=active_timeframes,
                )
            elif checkpoint_path and Path(checkpoint_path).exists():
                exported_base = export_forex_onnx(
                    checkpoint_path=checkpoint_path,
                    out_path=fp32_onnx_path,
                    active_timeframes=active_timeframes,
                )
            else:
                logger.warning("No model instance or checkpoint file provided for Forex ONNX tracing.")

            if exported_base and exported_base.exists():
                # Load the full FP32 model into memory before converting to sidecar
                m_fp32 = onnx.load(str(fp32_onnx_path))

                # Convert FP32 model to external data sidecar (.onnx.data)
                onnx.external_data_helper.convert_model_to_external_data(
                    m_fp32,
                    all_tensors_to_one_file=True,
                    location=fp32_data_filename,
                    size_threshold=1024,
                    convert_attribute=False,
                )
                onnx.save(m_fp32, str(fp32_onnx_path))
                exported_files["onnx_fp32"] = fp32_onnx_path
                exported_files["onnx_fp32_data"] = fp32_data_path
                logger.info(
                    "Exported Forex FP32 ONNX graph (%s) with binary data sidecar (%s)",
                    fp32_onnx_path.name,
                    fp32_data_filename,
                )

                # -----------------------------------------------------------------
                # 3. FP16 ONNX Model with Integrated Weights (.onnx)
                # -----------------------------------------------------------------
                fp16_onnx_path = target_dir / f"{model_key}.onnx"
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
        except Exception as exc:
            logger.error("Failed Forex ONNX tri-format export: %s", exc)

        return exported_files
