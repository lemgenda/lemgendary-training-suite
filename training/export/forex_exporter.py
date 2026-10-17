"""Forex model exporter for LemGendary AI Training Suite.

Handles MetaTrader 5 Expert Advisor signal ONNX and PyTorch checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.export.base import BaseModelExporter
from training.export.mt5_signal import export_forex_onnx
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.forex")


class ForexModelExporter(BaseModelExporter):
    """Exports multi-timeframe ForexPredictor models for MT5 integration."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if model is ForexPredictor."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            d_type = resolved_spec.dataset_type
            d_types = [str(x).lower() for x in d_type] if isinstance(d_type, list) else [str(d_type or "").lower()]
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
        """Export Forex model into ONNX and PyTorch checkpoint formats."""
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)

        checkpoint_path: Path | str | None = kwargs.get("checkpoint_path")
        active_timeframes: list[int] | None = kwargs.get("active_timeframes")
        if active_timeframes is None and resolved_spec:
            raw_cfg = resolved_spec.raw_config or {}
            active_timeframes = raw_cfg.get("kwargs", {}).get("active_timeframes", [1, 5, 15, 60, 240, 1440])

        exported_files: dict[str, Path] = {}

        # 1. Standalone PyTorch state dict (.pt)
        pt_path = target_dir / f"{model_key}.pt"
        try:
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), str(pt_path))
                exported_files["pt"] = pt_path
                logger.info("Saved Forex PyTorch model state dict to %s", pt_path)
        except Exception as exc:
            logger.warning("Failed saving Forex .pt state dict: %s", exc)

        # 2. MT5-compatible ONNX export
        onnx_path = target_dir / f"{model_key}.onnx"
        if checkpoint_path and Path(checkpoint_path).exists():
            exported_onnx = export_forex_onnx(
                checkpoint_path=checkpoint_path,
                out_path=onnx_path,
                active_timeframes=active_timeframes,
            )
            exported_files["onnx"] = exported_onnx
            exported_files["onnx_fp16"] = exported_onnx
        else:
            logger.warning("No checkpoint file provided for Forex ONNX tracing.")

        return exported_files
