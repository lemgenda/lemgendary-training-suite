"""
LemGendary MT5 Signal Generator & ONNX Exporter v1.0
======================================================
Post-training utilities for deploying the ForexPredictor to MetaTrader 5.

Two modes:
    1. ONNX Export:     Convert trained .pth checkpoint to .onnx for MT5 EA.
    2. Live Signal:     Connect to MT5, pull latest bars, run model, emit signal.

Usage (run as a module from the project root):
    # Export trained model to ONNX:
    python -m export.mt5_signal --mode export \
        --checkpoint LemGendaryModels/forex_predictor/forex_predictor_best.pth

    # Generate a live signal (requires MT5 terminal + demo account):
    python -m export.mt5_signal --mode signal \
        --onnx LemGendaryModels/forex_predictor/forex_predictor.onnx --pair EURUSD

    # Write the MQL5 Expert Advisor (rendered from the registry):
    python -m export.mt5_signal --mode mql5_ea --out export/LemGendaryForexEA.mq5

The MT5 model is the FP16 ``<Stem>.onnx`` (FP32 inputs/outputs, embedded weights, IR 9 for
the ONNX Runtime 1.17 bundled with MT5). Check it under that runtime with
``tools/mt5_ort_check.py`` and in the MT5 Strategy Tester before live use.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from data.forex_spec import PAIR_INDEX, TIMEFRAME_LOOKBACK
from data.mt5_feature_spec import WARMUP_BARS
from data.mt5_features import (
    FEATURES,
    compute_indicators,
    normalize_ohlcv,
)
from data.mt5_pipeline import MT5Credentials, add_timeframes_argument
from data.mt5_terminal import (
    connect_mt5,
    disconnect_mt5,
    download_bars,
)
from training.cloud.variations import ONNX_FP16_SUFFIX
from training.export.common import DEFAULT_ONNX_OPSET
from training.export.mql5_ea import ea_spec, render_mql5_ea
from training.export.mt5_signal import (
    DEFAULT_TIMEFRAMES,
    DIRECTION_LABELS,
    forex_variant_writer,
    load_forex_checkpoint,
)
from training.export.onnx_variants import ONNX_FP16_VARIANT, ONNX_FP32_VARIANT
from training.export.torch_standalone import export_torch_standalone
from training.export.validation import resolve_model_stem
from training.registry import get_model_registry
from training.utils.lazy_imports import load_onnxruntime

# [LemGendary Forex Suite v1.0 - SYNC_ID: FOREX_03]

logger = logging.getLogger("lemtrain.export.mt5_signal")

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────


def _export_paths(out_path: str) -> tuple[str, str, str]:
    """FP32 ONNX, FP16 ONNX and ``.pt`` paths next to ``out_path`` (``_FP32`` aware)."""
    base_dir = os.path.dirname(out_path)
    base_stem = os.path.splitext(os.path.basename(out_path))[0]
    fp32_onnx_path = (
        os.path.join(base_dir, f"{base_stem}_FP32.onnx")
        if not base_stem.endswith("_FP32")
        else out_path
    )
    fp16_onnx_path = (
        out_path
        if not base_stem.endswith("_FP32")
        else os.path.join(base_dir, f"{base_stem.replace('_FP32', '')}.onnx")
    )
    pt_path = os.path.join(base_dir, f"{base_stem.replace('_FP32', '')}.pt")
    return fp32_onnx_path, fp16_onnx_path, pt_path


def export_onnx(
    checkpoint_path: str,
    out_path: str | None = None,
    active_timeframes: list | None = None,
    opset: int = DEFAULT_ONNX_OPSET,
):
    """
    Export a trained ForexPredictor to ONNX format.

    The export uses a single representative batch to trace the model.
    All timeframe inputs are exported as separate named inputs for flexibility.

    Args:
        checkpoint_path:    Path to .pth checkpoint (best or latest).
        out_path:           Output .onnx file path. Defaults to same dir as checkpoint.
        active_timeframes:  Timeframe rungs to include (default: [60]).
        opset:              ONNX opset version (18, the torch.export exporter minimum).

    Returns:
        Path to the exported .onnx file.
    """
    if active_timeframes is None:
        active_timeframes = list(DEFAULT_TIMEFRAMES)

    if out_path is None:
        base = os.path.splitext(checkpoint_path)[0]
        out_path = base + ".onnx"

    model = load_forex_checkpoint(checkpoint_path, active_timeframes)
    fp32_onnx_path, fp16_onnx_path, pt_path = _export_paths(out_path)
    writer = forex_variant_writer(model, active_timeframes, opset)

    # Export 1: FP32 ONNX with its .onnx.data weights sidecar
    print(f" [ONNX Export] Synthesizing FP32 ONNX model to: {fp32_onnx_path} (opset {opset})")
    writer.write(ONNX_FP32_VARIANT, Path(fp32_onnx_path))

    # Export 2: FP16 ONNX for the EA: FP32 inputs/outputs, one file, IR 9 (MT5 ORT 1.17)
    print(f" [ONNX Export] Synthesizing production FP16 ONNX model to: {fp16_onnx_path}")
    writer.write(ONNX_FP16_VARIANT, Path(fp16_onnx_path))

    # Export 3: Standalone PyTorch FP32 model (.pt)
    print(f" [ONNX Export] Saving standalone PyTorch FP32 model to: {pt_path}")
    export_torch_standalone(model, pt_path, metadata={"model_key": "forex_predictor"})

    print(" [ONNX Export] [OK] Export Matrix Complete:")
    print(f"   -> FP32 ONNX:  {fp32_onnx_path}")
    print(f"   -> FP16 ONNX:  {fp16_onnx_path}")
    print(f"   -> PyTorch PT: {pt_path}")
    return fp16_onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Inference
# ─────────────────────────────────────────────────────────────────────────────


def load_onnx_session(onnx_path: str):
    """
    Load an ONNX model for inference.
    Uses DirectML on Windows, CUDA on Linux (matching requirements.txt config).
    """
    try:
        ort = load_onnxruntime()
    except ImportError as exc:
        raise RuntimeError("[ONNX] onnxruntime not installed. Check requirements.txt.") from exc

    providers = []
    if sys.platform == "win32":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f" [ONNX] Session loaded: {onnx_path}")
    print(f" [ONNX] Providers: {session.get_providers()}")
    return session


def run_onnx_inference(
    session,
    tf_inputs: dict,
    pair_idx: int,
) -> dict:
    """
    Run ONNX inference for a single bar.

    Args:
        session:   ONNX Runtime session.
        tf_inputs: Dict[int → np.ndarray[seq_len, features]]
        pair_idx:  Integer pair index (see PAIR_INDEX).

    Returns:
        Dict with 'direction' (str), 'confidence' (float), 'tp_pips', 'sl_pips'.
    """
    feed = {}
    for tf, arr in tf_inputs.items():
        feed[f"tf_{tf}"] = arr[np.newaxis].astype(np.float32)  # Add batch dim
    feed["pair_idx"] = np.array([pair_idx], dtype=np.int64)

    dir_logits, magnitude = session.run(None, feed)

    probs = _softmax(dir_logits[0])
    direction = int(np.argmax(probs))
    confidence = float(probs[direction])
    tp_pips = float(magnitude[0, 0])
    sl_pips = float(magnitude[0, 1])

    return {
        "direction": DIRECTION_LABELS[direction],
        "confidence": confidence,
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "probs": {"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])},
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Live Signal Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_signal(
    onnx_path: str,
    pair: str = "EURUSD",
    active_timeframes: list | None = None,
    credentials: MT5Credentials | None = None,
) -> dict:
    """
    Pull latest bars from MT5, preprocess, run ONNX model, return trade signal.

    Args:
        onnx_path:          Path to exported .onnx model file.
        pair:               Currency pair symbol.
        active_timeframes:  Timeframes to query (must match export).
        credentials:        MT5 demo account login, password and server (``None`` uses the
                            MT5_* environment variables).

    Returns:
        Signal dict: {direction, confidence, tp_pips, sl_pips, probs}
    """
    if active_timeframes is None:
        active_timeframes = list(DEFAULT_TIMEFRAMES)

    creds = credentials or MT5Credentials()
    if not connect_mt5(login=creds.login, password=creds.password, server=creds.server):
        return {
            "error": "MT5 not connected. Provide demo account credentials.",
            "help": (
                "Pass --login, --password, --server or set MT5_LOGIN, MT5_PASSWORD, "
                "MT5_SERVER environment variables."
            ),
        }

    try:
        session = load_onnx_session(onnx_path)
        p_idx = PAIR_INDEX.get(pair, 0)
        tf_inputs = {}

        for tf in active_timeframes:
            seq_len = TIMEFRAME_LOOKBACK[tf]
            # Pull enough bars to compute indicators + window (same warm-up as the EA)
            n_bars = seq_len + WARMUP_BARS

            df = download_bars(pair, tf, n_bars=n_bars)
            df = compute_indicators(df)
            df = normalize_ohlcv(df)

            # Every model input column in specification order (data.mt5_feature_spec).
            arr = df[FEATURES].to_numpy(dtype=np.float32)

            # Take last seq_len bars
            if len(arr) < seq_len:
                raise RuntimeError(
                    f"Insufficient bars for {pair}@{tf}min: got {len(arr)}, need {seq_len}"
                )
            tf_inputs[tf] = arr[-seq_len:]

        signal = run_onnx_inference(session, tf_inputs, p_idx)
        signal["pair"] = pair
        return signal

    finally:
        disconnect_mt5()


# ─────────────────────────────────────────────────────────────────────────────
# MQL5 Expert Advisor
# ─────────────────────────────────────────────────────────────────────────────

FOREX_MODEL_KEY = "forex_predictor"


def generate_mql5_ea(out_path: str, model_key: str = FOREX_MODEL_KEY) -> Path:
    """
    Write the MQL5 Expert Advisor for ``model_key`` (rendered from its registry entry).

    The EA loads the single-file FP16 release model (FP32 I/O, IR 9) with ``OnnxCreate``;
    ``RiskPct`` defaults to the registry ``evaluation.risk_per_trade``.

    Args:
        out_path: Path to write the .mq5 file.
        model_key: Registry key of the Forex model.
    """
    spec = get_model_registry().get_model(model_key)
    if spec is None:
        raise KeyError(f"Model {model_key!r} is not in the registry")
    model_file = f"{resolve_model_stem(model_key, spec)}{ONNX_FP16_SUFFIX}"
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_mql5_ea(ea_spec(spec.raw_config, model_file)), encoding="utf-8")
    print(f" [MQL5] Expert Advisor written: {target}")
    print(" [MQL5] Compile it in MetaEditor and run the Strategy Tester before live trading.")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entrypoint: ONNX export, live signal generation or MQL5 Expert Advisor writing."""
    parser = argparse.ArgumentParser(description="LemGendary MT5 Signal Generator & ONNX Exporter")
    parser.add_argument(
        "--mode",
        choices=["export", "signal", "mql5_ea"],
        default="export",
        help=(
            "export: convert to ONNX | signal: generate live signal | mql5_ea: write the MQL5 EA"
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint for export")
    parser.add_argument("--onnx", default=None, help="Path to .onnx model for signal mode")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair for signal mode")
    add_timeframes_argument(
        parser, "Active timeframe rungs in minutes (default: 1 5 15 60 240 1440)"
    )
    parser.add_argument("--out", default=None, help="Output path for ONNX or MQL5 file")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET, help="ONNX opset version")
    parser.add_argument("--login", type=int, default=None, help="MT5 demo account login ID")
    parser.add_argument("--password", type=str, default=None, help="MT5 demo account password")
    parser.add_argument("--server", type=str, default=None, help="MT5 demo account server name")
    args = parser.parse_args()

    if args.mode == "export":
        if not args.checkpoint:
            print("[ERROR] --checkpoint required for export mode.")
            print("[REMEDY] Provide a valid path to your PyTorch checkpoint using --checkpoint.")
            return
        export_onnx(
            checkpoint_path=args.checkpoint,
            out_path=args.out,
            active_timeframes=args.timeframes,
            opset=args.opset,
        )

    elif args.mode == "signal":
        if not args.onnx:
            print("[ERROR] --onnx required for signal mode.")
            print("[REMEDY] Provide a valid path to your ONNX model using --onnx.")
            return
        print(f" [Signal] Generating signal for {args.pair}...")
        result = generate_signal(
            onnx_path=args.onnx,
            pair=args.pair,
            active_timeframes=args.timeframes,
            credentials=MT5Credentials(
                login=args.login, password=args.password, server=args.server
            ),
        )
        print("\n [Signal] Result:")
        print(json.dumps(result, indent=2))

    elif args.mode == "mql5_ea":
        out = args.out or "export/LemGendaryForexEA.mq5"
        generate_mql5_ea(out)


if __name__ == "__main__":
    main()
