"""MetaTrader 5 Expert Advisor signal ONNX exporter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.export.common import ExportError

logger = logging.getLogger("lemtrain.export.mt5_signal")

TIMEFRAME_LOOKBACK: dict[int, int] = {
    1: 120,
    5: 72,
    15: 48,
    60: 48,
    240: 30,
    1440: 30,
}
FOREX_FEATURES_PER_BAR = 12


class _OnnxForexWrapper(nn.Module):
    """Wrapper to accept flat positional args for ONNX tracing."""

    def __init__(self, model: nn.Module, tf_keys: list[str]) -> None:
        super().__init__()
        self.model = model
        self.tf_keys = tf_keys

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx = args[-1]
        tf_tensors = args[:-1]
        tf_dict = {int(k): t for k, t in zip(self.tf_keys, tf_tensors)}
        out = self.model(tf_dict, pair_idx)
        return out["direction_logits"], out["magnitude"]


def export_forex_onnx(
    checkpoint_path: Path | str,
    out_path: Path | str | None = None,
    active_timeframes: list[int] | None = None,
    opset: int = 17,
) -> Path:
    """Export trained ForexPredictor checkpoint to MT5-compatible ONNX."""
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.exists():
        raise ExportError("CKPT_NOT_FOUND", f"Checkpoint not found at {ckpt_file}")

    timeframes = active_timeframes or [1, 5, 15, 60, 240, 1440]
    out_file = Path(out_path).resolve() if out_path else ckpt_file.with_suffix(".onnx")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading forex checkpoint from %s...", ckpt_file)
    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ExportError("LOAD_FAILED", f"Could not load checkpoint {ckpt_file}: {exc}") from exc

    try:
        from models.forex_predictor import ForexPredictor
    except ImportError as exc:
        raise ExportError("MODEL_DEF_MISSING", f"ForexPredictor class could not be imported: {exc}") from exc

    raw_kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
    model_kwargs: dict[str, Any] = dict(raw_kwargs)
    model_kwargs["active_timeframes"] = timeframes
    model = ForexPredictor(**model_kwargs)

    raw_state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in raw_state.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    batch_size = 1
    in_feats = getattr(model, "in_features", FOREX_FEATURES_PER_BAR)
    tf_inputs = [
        torch.zeros(batch_size, TIMEFRAME_LOOKBACK.get(tf, 48), in_feats)
        for tf in timeframes
    ]
    pair_dummy = torch.zeros(batch_size, dtype=torch.long)

    tf_keys = [str(tf) for tf in timeframes]
    wrapper = _OnnxForexWrapper(model, tf_keys)
    wrapper.eval()

    input_names = [f"tf_{tf}" for tf in timeframes] + ["pair_idx"]
    output_names = ["direction_logits", "magnitude"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    logger.info("Exporting Forex ONNX model to %s (opset=%d)...", out_file, opset)
    try:
        torch.onnx.export(
            wrapper,
            tuple(tf_inputs) + (pair_dummy,),
            str(out_file),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("Forex ONNX export successful: %s", out_file)
        return out_file
    except Exception as exc:
        err_msg = f"Forex ONNX export failed: {exc}"
        logger.error(err_msg)
        raise ExportError("FOREX_ONNX_EXPORT_FAILED", err_msg) from exc
