"""MetaTrader 5 Expert Advisor signal ONNX graph (ForexPredictor) and its export helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.forex_spec import TIMEFRAME_LOOKBACK
from models.forex_predictor import FOREX_FEATURES_PER_BAR, ForexPredictor
from training.export.base import EXPORT_ERRORS, MODEL_IO_ERRORS
from training.export.common import (
    DEFAULT_ONNX_OPSET,
    ExportError,
    OnnxGraphSpec,
    load_checkpoint_weights,
)
from training.export.onnx_variants import ONNX_FP32_VARIANT, OnnxVariantWriter

logger = logging.getLogger("lemtrain.export.mt5_signal")

DEFAULT_TIMEFRAMES: tuple[int, ...] = (1, 5, 15, 60, 240, 1440)
FOREX_OUTPUT_NAMES: tuple[str, ...] = ("direction_logits", "magnitude")
_SAMPLE_BATCH = 2
"""Export sample batch: ``torch.export`` specialises an axis whose sample size is 1."""


class _OnnxForexWrapper(nn.Module):
    """Wrapper to accept flat positional args for ONNX export."""

    def __init__(self, model: nn.Module, tf_keys: list[str]) -> None:
        super().__init__()
        self.model = model
        self.tf_keys = tf_keys

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map flat (timeframe tensors..., pair_idx) args to the model and return its two heads."""
        pair_idx = args[-1]
        tf_tensors = args[:-1]
        tf_dict = {int(k): t for k, t in zip(self.tf_keys, tf_tensors)}
        out = self.model(tf_dict, pair_idx)
        return out["direction_logits"], out["magnitude"]


def _dummy_inputs(model: nn.Module, timeframes: list[int]) -> tuple[torch.Tensor, ...]:
    """Zero tensors ``(tf_1, ..., tf_n, pair_idx)`` for the sample batch on the model's device."""
    in_feats = getattr(model, "in_features", FOREX_FEATURES_PER_BAR)
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    tf_inputs = [
        torch.zeros(_SAMPLE_BATCH, TIMEFRAME_LOOKBACK.get(tf, 48), in_feats, device=device)
        for tf in timeframes
    ]
    pair_dummy = torch.arange(_SAMPLE_BATCH, dtype=torch.long, device=device)
    return (*tf_inputs, pair_dummy)


@dataclass(frozen=True)
class ForexOnnxGraph:
    """The flat-argument Forex module, its export sample and its graph spec."""

    module: nn.Module
    inputs: tuple[torch.Tensor, ...]
    graph: OnnxGraphSpec


def forex_onnx_graph(
    model: nn.Module,
    active_timeframes: list[int] | None = None,
    opset: int = DEFAULT_ONNX_OPSET,
) -> ForexOnnxGraph:
    """Wrap ``model`` for ONNX: one ``tf_<minutes>`` input per timeframe plus ``pair_idx``."""
    timeframes = list(
        active_timeframes or getattr(model, "active_timeframes", None) or DEFAULT_TIMEFRAMES
    )
    model.eval()
    wrapper = _OnnxForexWrapper(model, [str(tf) for tf in timeframes]).eval()
    input_names = [f"tf_{tf}" for tf in timeframes] + ["pair_idx"]
    graph = OnnxGraphSpec(
        input_names=input_names,
        output_names=list(FOREX_OUTPUT_NAMES),
        dynamic_axes={name: {0: "batch"} for name in input_names},
        opset=opset,
    )
    return ForexOnnxGraph(wrapper, _dummy_inputs(model, timeframes), graph)


def forex_variant_writer(
    model: nn.Module,
    active_timeframes: list[int] | None = None,
    opset: int = DEFAULT_ONNX_OPSET,
) -> OnnxVariantWriter:
    """Variant writer for a ForexPredictor; reduced-precision variants keep FP32 I/O for MT5."""
    fx = forex_onnx_graph(model, active_timeframes, opset)
    return OnnxVariantWriter(fx.module, fx.inputs, fx.graph, fp32_io=True)


def export_forex_onnx_from_model(
    model: nn.Module,
    out_path: Path | str,
    active_timeframes: list[int] | None = None,
    opset: int = DEFAULT_ONNX_OPSET,
) -> Path:
    """Export an in-memory ForexPredictor to an MT5-compatible FP32 ONNX graph (+ sidecar)."""
    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fx = forex_onnx_graph(model, active_timeframes, opset)
    logger.info("Exporting Forex ONNX model to %s (opset=%d)...", out_file, opset)
    try:
        OnnxVariantWriter(fx.module, fx.inputs, fx.graph).write(ONNX_FP32_VARIANT, out_file)
    except EXPORT_ERRORS as exc:
        err_msg = f"Forex ONNX export failed: {exc}"
        logger.error(err_msg)
        raise ExportError("FOREX_ONNX_EXPORT_FAILED", err_msg) from exc
    logger.info("Forex ONNX export successful: %s", out_file)
    return out_file


def load_forex_checkpoint(
    checkpoint_path: Path | str, active_timeframes: list[int] | None = None
) -> ForexPredictor:
    """Rebuild a ForexPredictor from a checkpoint with its weights loaded (eval mode)."""
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.exists():
        raise ExportError("CKPT_NOT_FOUND", f"Checkpoint not found at {ckpt_file}")
    logger.info("Loading forex checkpoint from %s...", ckpt_file)
    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    except MODEL_IO_ERRORS as exc:
        raise ExportError("LOAD_FAILED", f"Could not load checkpoint {ckpt_file}: {exc}") from exc

    raw_kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
    model_kwargs: dict[str, Any] = dict(raw_kwargs)
    model_kwargs["active_timeframes"] = list(active_timeframes or DEFAULT_TIMEFRAMES)
    model = ForexPredictor(**model_kwargs)
    load_checkpoint_weights(model, ckpt)
    model.eval()
    return model


def export_forex_onnx(
    checkpoint_path: Path | str,
    out_path: Path | str | None = None,
    active_timeframes: list[int] | None = None,
    opset: int = DEFAULT_ONNX_OPSET,
) -> Path:
    """Export trained ForexPredictor checkpoint to an MT5-compatible FP32 ONNX graph."""
    timeframes = list(active_timeframes or DEFAULT_TIMEFRAMES)
    model = load_forex_checkpoint(checkpoint_path, timeframes)
    ckpt_file = Path(checkpoint_path).resolve()
    out_file = Path(out_path).resolve() if out_path else ckpt_file.with_suffix(".onnx")
    return export_forex_onnx_from_model(
        model=model,
        out_path=out_file,
        active_timeframes=timeframes,
        opset=opset,
    )
