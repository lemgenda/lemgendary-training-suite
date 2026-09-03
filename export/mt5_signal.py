"""
LemGendary MT5 Signal Generator & ONNX Exporter v1.0
======================================================
Post-training utilities for deploying the ForexPredictor to MetaTrader 5.

Two modes:
    1. ONNX Export:     Convert trained .pth checkpoint to .onnx for MT5 EA.
    2. Live Signal:     Connect to MT5, pull latest bars, run model, emit signal.

Usage:
    # Export trained model to ONNX:
    python export/mt5_signal.py --mode export --checkpoint LemGendaryModels/forex_predictor/forex_predictor_best.pth

    # Generate a live signal (requires MT5 terminal + demo account):
    python export/mt5_signal.py --mode signal --onnx LemGendaryModels/forex_predictor/forex_predictor.onnx --pair EURUSD

TODO (requires MT5 demo account):
    - Validate ONNX inference matches PyTorch output on real bars
    - Wire MT5 live bar pull into signal() function
    - Test MQL5 EA stub in MT5 Strategy Tester
"""

import os
import sys
import argparse
import json
import numpy as np

# Anchor workspace root for relative imports
_script_dir    = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(_script_dir)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from data.mt5_pipeline import (
    TIMEFRAME_LOOKBACK,
    MAJOR_PAIRS,
    PAIR_INDEX,
    compute_indicators,
    normalize_ohlcv,
    generate_labels,
    connect_mt5,
    disconnect_mt5,
    download_bars,
)
from models.forex_predictor import ForexPredictor, DIRECTION_CLASSES

# [LemGendary Forex Suite v1.0 - SYNC_ID: FOREX_03]

DIRECTION_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    checkpoint_path: str,
    out_path: str | None = None,
    active_timeframes: list | None = None,
    opset: int = 17,
):
    """
    Export a trained ForexPredictor to ONNX format.

    The export uses a single representative batch to trace the model.
    All timeframe inputs are exported as separate named inputs for flexibility.

    Args:
        checkpoint_path:    Path to .pth checkpoint (best or latest).
        out_path:           Output .onnx file path. Defaults to same dir as checkpoint.
        active_timeframes:  Timeframe rungs to include (default: [60]).
        opset:              ONNX opset version (17 recommended).

    Returns:
        Path to the exported .onnx file.
    """
    import torch
    import torch.onnx

    if active_timeframes is None:
        active_timeframes = [1, 5, 15, 60, 240, 1440]

    if out_path is None:
        base = os.path.splitext(checkpoint_path)[0]
        out_path = base + ".onnx"

    print(f" [ONNX Export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct model from checkpoint kwargs or defaults
    model_kwargs = ckpt.get("model_kwargs", {})
    model_kwargs["active_timeframes"] = active_timeframes
    model = ForexPredictor(**model_kwargs)

    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    # Strip DataParallel prefix if present
    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Build dummy inputs for tracing
    B = 1
    tf_inputs_dummy = {
        tf: torch.zeros(B, TIMEFRAME_LOOKBACK[tf], 10)
        for tf in active_timeframes
    }
    pair_dummy = torch.zeros(B, dtype=torch.long)

    # ONNX tracing requires flat inputs — flatten tf_inputs dict to args
    tf_keys       = [str(tf) for tf in active_timeframes]
    flat_tf_list  = [tf_inputs_dummy[tf] for tf in active_timeframes]

    class _OnnxWrapper(torch.nn.Module):
        """Wrapper to accept flat positional args for ONNX tracing."""
        def __init__(self, model, tf_keys):
            super().__init__()
            self.model   = model
            self.tf_keys = tf_keys

        def forward(self, *args):
            pair_idx    = args[-1]
            tf_tensors  = args[:-1]
            tf_dict     = {int(k): t for k, t in zip(self.tf_keys, tf_tensors)}
            out         = self.model(tf_dict, pair_idx)
            return out["direction_logits"], out["magnitude"]

    wrapper = _OnnxWrapper(model, tf_keys)
    wrapper.eval()

    input_names  = [f"tf_{tf}" for tf in active_timeframes] + ["pair_idx"]
    output_names = ["direction_logits", "magnitude"]

    # Export 1: FP32 ONNX
    base_dir = os.path.dirname(out_path)
    base_stem = os.path.splitext(os.path.basename(out_path))[0]
    fp32_onnx_path = os.path.join(base_dir, f"{base_stem}_FP32.onnx") if not base_stem.endswith("_FP32") else out_path
    fp16_onnx_path = out_path if not base_stem.endswith("_FP32") else os.path.join(base_dir, f"{base_stem.replace('_FP32', '')}.onnx")
    pt_path = os.path.join(base_dir, f"{base_stem.replace('_FP32', '')}.pt")

    print(f" [ONNX Export] Synthesizing FP32 ONNX model to: {fp32_onnx_path} (opset {opset})")
    torch.onnx.export(
        wrapper,
        tuple(flat_tf_list) + (pair_dummy,),
        fp32_onnx_path,
        input_names   = input_names,
        output_names  = output_names,
        dynamic_axes  = {name: {0: "batch"} for name in input_names + output_names},
        opset_version = opset,
        do_constant_folding = True,
    )

    try:
        import onnxsim
        import onnx
        model_onnx = onnx.load(fp32_onnx_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simplified, fp32_onnx_path)
            print(" [ONNX Export] Model simplified successfully via onnxsim.")
    except Exception as e:
        print(f" [ONNX Export] onnxsim skipped: {e}")

    # Export 2: FP16 ONNX with embedded weights
    print(f" [ONNX Export] Synthesizing production FP16 ONNX model to: {fp16_onnx_path}")
    try:
        import onnx
        from onnxconverter_common import float16
        model_fp32 = onnx.load(fp32_onnx_path)
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
        onnx.save(model_fp16, fp16_onnx_path)
        print(" [ONNX Export] [OK] FP16 embedded weights ONNX model saved.")
    except Exception as e:
        print(f" [ONNX Export] FP16 conversion fallback: copying FP32 -> {fp16_onnx_path}")
        import shutil
        shutil.copy2(fp32_onnx_path, fp16_onnx_path)

    # Detach FP32 weights into external sidecar file (.onnx.data)
    try:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        fp32_filename = os.path.basename(fp32_onnx_path)
        sidecar_filename = f"{fp32_filename}.data"
        sidecar_path = os.path.join(base_dir, sidecar_filename)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
            
        onnx_model_fp32 = onnx.load(fp32_onnx_path)
        convert_model_to_external_data(
            onnx_model_fp32,
            all_tensors_to_one_file=True,
            location=sidecar_filename,
            size_threshold=1024
        )
        onnx.save(onnx_model_fp32, fp32_onnx_path)
        print(f" [ONNX Export] [OK] Detached FP32 weights to external sidecar: {sidecar_filename}")
    except Exception as e:
        print(f" [ONNX Export] External data detachment skipped: {e}")

    # Export 3: Standalone PyTorch FP32 model object (.pt)
    print(f" [ONNX Export] Saving standalone PyTorch FP32 model to: {pt_path}")
    torch.save(model, pt_path)

    print(f" [ONNX Export] [OK] Export Matrix Complete:")
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
        import onnxruntime as ort
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

    probs      = _softmax(dir_logits[0])
    direction  = int(np.argmax(probs))
    confidence = float(probs[direction])
    tp_pips    = float(magnitude[0, 0])
    sl_pips    = float(magnitude[0, 1])

    return {
        "direction":  DIRECTION_LABELS[direction],
        "confidence": confidence,
        "tp_pips":    tp_pips,
        "sl_pips":    sl_pips,
        "probs":      {"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])},
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
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
) -> dict:
    """
    Pull latest bars from MT5, preprocess, run ONNX model, return trade signal.

    Args:
        onnx_path:          Path to exported .onnx model file.
        pair:               Currency pair symbol.
        active_timeframes:  Timeframes to query (must match export).
        login:              MT5 demo account login ID.
        password:           MT5 demo account password.
        server:             MT5 demo account server name.

    Returns:
        Signal dict: {direction, confidence, tp_pips, sl_pips, probs}
    """
    if active_timeframes is None:
        active_timeframes = [1, 5, 15, 60, 240, 1440]

    if not connect_mt5(login=login, password=password, server=server):
        return {
            "error": "MT5 not connected. Provide demo account credentials.",
            "help":  "Pass --login, --password, --server or set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER environment variables.",
        }

    try:
        session  = load_onnx_session(onnx_path)
        p_idx    = PAIR_INDEX.get(pair, 0)
        tf_inputs = {}

        for tf in active_timeframes:
            seq_len = TIMEFRAME_LOOKBACK[tf]
            # Pull enough bars to compute indicators + window
            n_bars  = seq_len + 300  # Extra for indicator warmup

            df = download_bars(pair, tf, n_bars=n_bars)
            df = compute_indicators(df)
            df = normalize_ohlcv(df)

            features = ["open", "high", "low", "close", "volume",
                        "rsi", "macd", "macd_signal", "atr", "bb_width"]
            avail    = [c for c in features if c in df.columns]
            arr      = df[avail].values.astype(np.float32)

            # Take last seq_len bars
            if len(arr) < seq_len:
                raise RuntimeError(f"Insufficient bars for {pair}@{tf}min: got {len(arr)}, need {seq_len}")
            tf_inputs[tf] = arr[-seq_len:]

        signal = run_onnx_inference(session, tf_inputs, p_idx)
        signal["pair"] = pair
        return signal

    finally:
        disconnect_mt5()


# ─────────────────────────────────────────────────────────────────────────────
# MQL5 EA Stub Generator
# ─────────────────────────────────────────────────────────────────────────────

MQL5_EA_STUB = '''
//+------------------------------------------------------------------+
//| LemGendary Forex EA Stub                                         |
//| Generated by export/mt5_signal.py                               |
//| TODO: Wire to Python signal bridge before live deployment        |
//+------------------------------------------------------------------+
#property copyright "LemGendary AI"
#property version   "1.00"

// ── Safety Parameters ────────────────────────────────────────────────
input double   RiskPct        = 1.0;    // Risk per trade (% of balance)
input double   MinConfidence  = 0.65;   // Minimum model confidence to trade
input int      MagicNumber    = 20260724;
input string   PythonScript   = "lemgendary_signal.py"; // Signal bridge script

// ── Internal State ────────────────────────────────────────────────────
int    lastBar = 0;
double tp_pips, sl_pips;
string direction;
double confidence;

//+------------------------------------------------------------------+
void OnTick() {
    int currentBar = (int)(TimeCurrent() / PeriodSeconds());
    if (currentBar == lastBar) return;   // Only act on new bars
    lastBar = currentBar;

    // ── Call Python Signal Bridge ──────────────────────────────────
    // TODO: Implement IPC or named pipe bridge to mt5_signal.py
    // The bridge returns JSON: {"direction":"BUY","confidence":0.72,"tp_pips":35.0,"sl_pips":18.0}
    //
    // Example bridge call (pseudo-code):
    //   string json = CallPythonBridge(PythonScript, Symbol());
    //   ParseSignal(json, direction, confidence, tp_pips, sl_pips);

    // ── Safety Gate ───────────────────────────────────────────────
    // if (confidence < MinConfidence) return;   // Model not confident — skip bar

    // ── Position Sizing ───────────────────────────────────────────
    // double lots = CalcLotSize(RiskPct, sl_pips);  // Risk-based lot sizing

    // ── Execute ───────────────────────────────────────────────────
    // if (direction == "BUY")  OrderSend(Symbol(), OP_BUY,  lots, Ask, 3, Ask-sl_pips*Point, Ask+tp_pips*Point, "", MagicNumber);
    // if (direction == "SELL") OrderSend(Symbol(), OP_SELL, lots, Bid, 3, Bid+sl_pips*Point, Bid-tp_pips*Point, "", MagicNumber);
}
//+------------------------------------------------------------------+
'''


def generate_mql5_ea(out_path: str):
    """
    Write the MQL5 EA stub to disk.

    Args:
        out_path: Path to write the .mq5 file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(MQL5_EA_STUB)
    print(f" [MQL5] EA stub written: {out_path}")
    print(" [MQL5] TODO: Implement Python bridge IPC before live deployment.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LemGendary MT5 Signal Generator & ONNX Exporter")
    parser.add_argument("--mode", choices=["export", "signal", "mql5_stub"], default="export",
                        help="export: convert to ONNX | signal: generate live signal | mql5_stub: write EA template")
    parser.add_argument("--checkpoint",   default=None,     help="Path to .pth checkpoint for export")
    parser.add_argument("--onnx",         default=None,     help="Path to .onnx model for signal mode")
    parser.add_argument("--pair",         default="EURUSD", help="Currency pair for signal mode")
    parser.add_argument("--timeframes",   nargs="+", type=int, default=[1, 5, 15, 60, 240, 1440],
                        help="Active timeframe rungs in minutes (default: 1 5 15 60 240 1440)")
    parser.add_argument("--out",          default=None,     help="Output path for ONNX or MQL5 file")
    parser.add_argument("--opset",        type=int, default=17, help="ONNX opset version")
    parser.add_argument("--login",       type=int, default=None, help="MT5 demo account login ID")
    parser.add_argument("--password",    type=str, default=None, help="MT5 demo account password")
    parser.add_argument("--server",      type=str, default=None, help="MT5 demo account server name")
    args = parser.parse_args()

    if args.mode == "export":
        if not args.checkpoint:
            print("[ERROR] --checkpoint required for export mode.")
            print("[REMEDY] Provide a valid path to your PyTorch checkpoint using --checkpoint.")
            return
        export_onnx(
            checkpoint_path  = args.checkpoint,
            out_path         = args.out,
            active_timeframes= args.timeframes,
            opset            = args.opset,
        )

    elif args.mode == "signal":
        if not args.onnx:
            print("[ERROR] --onnx required for signal mode.")
            print("[REMEDY] Provide a valid path to your ONNX model using --onnx.")
            return
        print(f" [Signal] Generating signal for {args.pair}...")
        result = generate_signal(
            onnx_path         = args.onnx,
            pair              = args.pair,
            active_timeframes = args.timeframes,
            login             = args.login,
            password          = args.password,
            server            = args.server,
        )
        print("\n [Signal] Result:")
        print(json.dumps(result, indent=2))

    elif args.mode == "mql5_stub":
        out = args.out or "export/LemGendaryForexEA.mq5"
        generate_mql5_ea(out)


if __name__ == "__main__":
    main()
