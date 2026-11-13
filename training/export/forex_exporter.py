"""Forex model exporter for LemGendary AI Training Suite.

Writes the three release variants of a ForexPredictor (MetaTrader 5 Expert Advisor signal):
``<Stem>.pt``, ``<Stem>_FP32.onnx`` + ``.onnx.data`` and ``<Stem>.onnx`` (FP16 weights with
FP32 inputs and outputs, so the EA can feed float arrays to either graph).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch import nn

from models.forex_predictor import ForexPredictor
from training.cloud.variations import PT_SUFFIX
from training.export.base import (
    EXPORT_ERRORS,
    MODEL_IO_ERRORS,
    BaseModelExporter,
    spec_dataset_types,
)
from training.export.mt5_signal import (
    DEFAULT_TIMEFRAMES,
    forex_variant_writer,
    load_forex_checkpoint,
)
from training.export.onnx_variants import write_release_variants
from training.export.torch_standalone import export_torch_standalone
from training.export.validation import resolve_model_stem
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.forex")


def _resolve_forex_model(
    model: Any, checkpoint_path: Path | str | None, active_timeframes: list[int]
) -> nn.Module | None:
    """The in-memory model, else one rebuilt from ``checkpoint_path``; None if neither."""
    if isinstance(model, nn.Module):
        return model.eval()
    if checkpoint_path and Path(checkpoint_path).exists():
        return load_forex_checkpoint(checkpoint_path, active_timeframes)
    logger.warning("No model instance or checkpoint file provided for Forex export.")
    return None


def _pt_metadata(model_key: str, model: nn.Module) -> dict[str, Any]:
    """``.pt`` metadata: the model key and, for a ``ForexPredictor``, its ``model_kwargs``.

    :func:`training.export.mt5_signal.load_forex_checkpoint` rebuilds the model from them, so a
    non-default architecture loads without the registry.
    """
    metadata: dict[str, Any] = {"model_key": model_key}
    if isinstance(model, ForexPredictor):
        metadata["model_kwargs"] = model.model_kwargs()
    return metadata


def _active_timeframes(requested: list[int] | None, spec: ModelSpecification | None) -> list[int]:
    """``requested``, else the registry kwargs' ``active_timeframes``, else the defaults."""
    if requested is None and spec:
        requested = (spec.raw_config or {}).get("kwargs", {}).get("active_timeframes")
    return list(requested or DEFAULT_TIMEFRAMES)


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
        """Export a Forex model (``model`` or ``checkpoint_path`` kwarg) to the three variants."""
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)
        stem = resolve_model_stem(model_key, resolved_spec)

        timeframes = _active_timeframes(kwargs.get("active_timeframes"), resolved_spec)

        exported: dict[str, Path] = {}
        try:
            forex_model = _resolve_forex_model(model, kwargs.get("checkpoint_path"), timeframes)
        except (*EXPORT_ERRORS, *MODEL_IO_ERRORS) as exc:
            logger.error("Could not load the Forex model for export: %s", exc)
            return exported
        if forex_model is None:
            return exported

        pt_path = target_dir / f"{stem}{PT_SUFFIX}"
        try:
            export_torch_standalone(
                forex_model, pt_path, metadata=_pt_metadata(model_key, forex_model)
            )
            exported["pt"] = pt_path
        except (*EXPORT_ERRORS, *MODEL_IO_ERRORS) as exc:
            logger.warning("Failed saving Forex .pt model: %s", exc)

        writer = forex_variant_writer(forex_model, timeframes)
        exported.update(write_release_variants(writer, target_dir, stem))
        return exported
