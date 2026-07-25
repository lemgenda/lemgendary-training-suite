"""
LemGendary MT5 Data Pipeline v1.0
===================================
Connects to MetaTrader 5, downloads OHLCV candles for all major pairs and
timeframes, computes technical indicators, generates direction+magnitude labels,
and serializes windowed samples as .npy shards for the ForexDataset.

Usage (requires MetaTrader 5 terminal running locally):
    python data/mt5_pipeline.py --mode download --pairs EURUSD GBPUSD --timeframes 60 240
    python data/mt5_pipeline.py --mode build_dataset --out_dir data/forex

TODO (requires demo account):
    - mt5.initialize() → connect to MT5 terminal
    - Validate session credentials via mt5.account_info()
    - Run full multi-pair download
    - Generate dataset shards
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirrors forex_predictor.py)
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_LOOKBACK = {
    1:    512,
    5:    288,
    15:   192,
    60:   168,
    240:   90,
    1440:  252,
}

# MT5 timeframe constants (set once MT5 is available)
MT5_TIMEFRAMES = {
    1:    "TIMEFRAME_M1",
    5:    "TIMEFRAME_M5",
    15:   "TIMEFRAME_M15",
    60:   "TIMEFRAME_H1",
    240:  "TIMEFRAME_H4",
    1440: "TIMEFRAME_D1",
}

MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD"]

# Ordered list of timeframe rungs in minutes (Governor curriculum ladder)
TIMEFRAME_RUNGS = [1, 5, 15, 60, 240, 1440]

# Currency pair index (0-based) — shared with forex_predictor.py
PAIR_INDEX = {
    "EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "USDCHF": 3,
    "AUDUSD": 4, "USDCAD": 5, "NZDUSD": 6, "XAUUSD": 7,
}

FEATURES = ["open", "high", "low", "close", "volume",  # OHLCV (5)
            "rsi", "macd", "macd_signal", "atr", "bb_width"]  # Indicators (5)

# Label generation: forward-looking window to measure price movement
LABEL_HORIZON_BARS = 20       # Look N bars ahead for TP/SL determination
DIRECTION_THRESHOLD_PIPS = 10 # Minimum move (pips) to be classified as Up/Down (else Sideways)

# ─────────────────────────────────────────────────────────────────────────────
# MT5 Connection (requires live MT5 terminal)
# ─────────────────────────────────────────────────────────────────────────────

def connect_mt5(login: int | None = None, password: str | None = None, server: str | None = None, api_key: str | None = None) -> bool:
    """
    Initialize MetaTrader 5 connection.

    Credentials can be passed explicitly, or fall back to environment variables:
        MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_API_KEY

    Returns:
        True if connection succeeded, False otherwise.
    """
    if login is None and "MT5_LOGIN" in os.environ:
        try:
            login = int(os.environ["MT5_LOGIN"])
        except ValueError:
            pass
    if password is None:
        password = os.environ.get("MT5_PASSWORD")
    if server is None:
        server = os.environ.get("MT5_SERVER")
    if api_key is None:
        api_key = os.environ.get("MT5_API_KEY")

    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
    except ImportError:
        print(" [MT5] MetaTrader5 package not installed. Run: pip install MetaTrader5")
        print(" [MT5] NOTE: MT5 package is Windows-only and requires MT5 terminal installed.")
        return False

    # First try attaching directly to the running MT5 terminal session
    if mt5.initialize():
        info = mt5.account_info()
        if info:
            print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
            return True

    # Fallback to explicit login parameters if active session not found
    init_kwargs = {}
    if login and password and server:
        init_kwargs = {"login": login, "password": password, "server": server}

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        if login and password and server:
            if mt5.initialize():
                if mt5.login(login, password=password, server=server):
                    info = mt5.account_info()
                    if info:
                        print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
                        return True
        print(f" [MT5] Initialize failed: {err}")
        print(" [MT5] Ensure MetaTrader 5 desktop application is open and logged into your account.")
        return False

    info = mt5.account_info()
    if info:
        print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
        return True

    print(" [MT5] Connected to MT5 terminal session.")
    return True


def disconnect_mt5():
    """Cleanly shut down the MT5 connection."""
    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
        mt5.shutdown()
        print(" [MT5] Disconnected.")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Download
# ─────────────────────────────────────────────────────────────────────────────

def download_bars(pair: str, timeframe_min: int, n_bars: int = 50000) -> pd.DataFrame:
    """
    Download historical OHLCV bars from MT5 for a given pair and timeframe.

    TODO: Enable this once MT5 demo account is connected.

    Args:
        pair:          Symbol (e.g. "EURUSD")
        timeframe_min: Timeframe in minutes (1, 5, 15, 60, 240, 1440)
        n_bars:        Number of bars to download

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
        tf_attr = MT5_TIMEFRAMES.get(timeframe_min)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_min}min")

        tf = getattr(mt5, tf_attr)
        rates = mt5.copy_rates_from_pos(pair, tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise RuntimeError(f"No data returned for {pair} {timeframe_min}min: {error}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(
            columns={'tick_volume': 'volume'}
        )
        print(f" [MT5] Downloaded {len(df)} bars for {pair} {timeframe_min}min")
        return df

    except ImportError:
        raise RuntimeError(
            "[MT5] MetaTrader5 not installed. Run: pip install MetaTrader5\n"
            "      Ensure MT5 terminal is running with a demo account."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators (pandas-ta backed, no MT5 dependency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.
    Uses pandas-ta if available, falls back to manual NumPy implementations.

    Indicators added:
        rsi        — RSI(14)
        macd       — MACD line (12-26-9)
        macd_signal— MACD signal line
        atr        — ATR(14), normalized by close price
        bb_width   — Bollinger Band width (20, 2σ), normalized by close price

    Args:
        df: DataFrame with [open, high, low, close, volume] columns.

    Returns:
        DataFrame with 5 additional indicator columns.
    """
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    high  = np.asarray(df['high'],  dtype=np.float64)
    low   = np.asarray(df['low'],   dtype=np.float64)

    try:
        import pandas_ta as ta  # type: ignore
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        rsi_col    = next((c for c in df.columns if c.upper().startswith('RSI_')), None)
        macd_col   = next((c for c in df.columns if 'MACD_' in c.upper() and 'H' not in c.upper() and 'S' not in c.upper()), None)
        macds_col  = next((c for c in df.columns if 'MACDS_' in c.upper()), None)
        atr_col    = next((c for c in df.columns if c.upper().startswith('ATRR_') or c.upper().startswith('ATR')), None)
        bbu_col    = next((c for c in df.columns if 'BBU_' in c.upper()), None)
        bbl_col    = next((c for c in df.columns if 'BBL_' in c.upper()), None)

        if rsi_col:
            df['rsi']  = (np.asarray(df[rsi_col], dtype=np.float64) / 100.0).tolist()
        else:
            df['rsi']  = 0.5
        if macd_col:
            df['macd'] = (np.asarray(df[macd_col], dtype=np.float64) / (close + 1e-8)).tolist()
        else:
            df['macd'] = 0.0
        if macds_col:
            df['macd_signal'] = (np.asarray(df[macds_col], dtype=np.float64) / (close + 1e-8)).tolist()
        else:
            df['macd_signal'] = 0.0
        if atr_col:
            df['atr'] = (np.asarray(df[atr_col], dtype=np.float64) / (close + 1e-8)).tolist()
        else:
            df['atr'] = 0.0
        if bbu_col and bbl_col:
            df['bb_width'] = ((np.asarray(df[bbu_col], dtype=np.float64) - np.asarray(df[bbl_col], dtype=np.float64)) / (close + 1e-8)).tolist()
        else:
            df['bb_width'] = 0.0

    except ImportError:
        warnings.warn("[MT5Pipeline] pandas-ta not found. Computing indicators manually.")
        df['rsi']         = _rsi_manual(close, 14).tolist()
        macd_line, sig    = _macd_manual(close, 12, 26, 9)
        df['macd']        = (macd_line / (close + 1e-8)).tolist()
        df['macd_signal'] = (sig       / (close + 1e-8)).tolist()
        df['atr']         = (_atr_manual(high, low, close, 14) / (close + 1e-8)).tolist()
        df['bb_width']    = (_bbwidth_manual(close, 20, 2)     / (close + 1e-8)).tolist()

    return df


def _rsi_manual(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss  = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_g = np.asarray(pd.Series(gain.tolist()).ewm(alpha=1/period, min_periods=period).mean(), dtype=np.float64)
    avg_l = np.asarray(pd.Series(loss.tolist()).ewm(alpha=1/period, min_periods=period).mean(), dtype=np.float64)
    rs    = avg_g / (avg_l + 1e-8)
    return np.asarray(1.0 - 1.0 / (1.0 + rs), dtype=np.float64)


def _ema_manual(arr: np.ndarray, span: int) -> np.ndarray:
    return np.asarray(pd.Series(arr.tolist()).ewm(span=span, adjust=False).mean(), dtype=np.float64)


def _macd_manual(close: np.ndarray, fast: int, slow: int, signal: int):
    macd_line = _ema_manual(close, fast) - _ema_manual(close, slow)
    sig_line  = _ema_manual(macd_line, signal)
    return macd_line, sig_line


def _atr_manual(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.asarray(pd.Series(tr.tolist()).ewm(span=period, adjust=False).mean(), dtype=np.float64)


def _bbwidth_manual(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    s     = pd.Series(close.tolist())
    mid   = s.rolling(period).mean()
    std   = s.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return np.asarray((upper - lower), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_labels(df: pd.DataFrame, pair: str, horizon: int = LABEL_HORIZON_BARS,
                    threshold_pips: float = DIRECTION_THRESHOLD_PIPS) -> pd.DataFrame:
    """
    Generate direction + magnitude labels for each bar.

    Direction (3-class):
        0 = Down  (close fell > threshold_pips within horizon)
        1 = Sideways (move < threshold_pips)
        2 = Up    (close rose > threshold_pips within horizon)

    Magnitude:
        tp_pips = peak upward move within horizon (pips)
        sl_pips = peak downward move within horizon (pips)

    Pip size per pair (standard 4-decimal vs JPY/XAU):
        USDJPY, XAUUSD → 0.01 pip size; others → 0.0001 pip size
    """
    pip_size = 0.01 if any(x in pair for x in ["JPY", "XAU"]) else 0.0001

    close_arr = np.asarray(df['close'], dtype=np.float64)
    high_arr  = np.asarray(df['high'],  dtype=np.float64)
    low_arr   = np.asarray(df['low'],   dtype=np.float64)
    n         = len(df)

    directions = np.ones(n, dtype=np.int64)   # Default: Sideways
    tp_pips    = np.zeros(n, dtype=np.float32)
    sl_pips    = np.zeros(n, dtype=np.float32)

    for i in range(n - horizon):
        entry     = close_arr[i]
        fut_high  = high_arr[i+1 : i+1+horizon].max()
        fut_low   = low_arr[i+1  : i+1+horizon].min()
        up_move   = (fut_high - entry) / pip_size
        down_move = (entry - fut_low)  / pip_size

        tp_pips[i] = float(up_move)
        sl_pips[i] = float(down_move)

        if up_move >= threshold_pips and up_move > down_move:
            directions[i] = 2  # Up
        elif down_move >= threshold_pips and down_move > up_move:
            directions[i] = 0  # Down
        # else: 1 (Sideways, already default)

    df = df.copy()
    df['direction'] = directions.tolist()
    df['tp_pips']   = tp_pips.tolist()
    df['sl_pips']   = sl_pips.tolist()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV to returns and volume z-score.

    - open/high/low/close → log-return relative to previous close (removes price-level bias)
    - volume              → z-score over rolling 50-bar window
    - Indicators          → already normalized during compute_indicators
    """
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    prev  = np.roll(close, 1)
    prev[0] = close[0]

    for col in ['open', 'high', 'low', 'close']:
        df[col] = np.log(np.asarray(df[col], dtype=np.float64) / (prev + 1e-8)).tolist()

    vol   = np.asarray(df['volume'], dtype=np.float64)
    vol_s = pd.Series(vol.tolist())
    df['volume'] = np.asarray((vol_s - vol_s.rolling(50).mean()) / (vol_s.rolling(50).std() + 1e-8), dtype=np.float64).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Shard Serialization
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, seq_len: int) -> tuple:
    """
    Slice the processed DataFrame into (features, direction, magnitude) windows.

    Args:
        df:      Processed DataFrame with FEATURES + direction/tp_pips/sl_pips columns.
        seq_len: Lookback window size.

    Returns:
        Tuple of (X, y_dir, y_mag):
            X:     [N, seq_len, n_features]  float32
            y_dir: [N]                        int64 direction class
            y_mag: [N, 2]                     float32 (tp_pips, sl_pips)
    """
    available = [c for c in FEATURES if c in df.columns]
    feat_arr  = np.asarray(df[available], dtype=np.float32)    # [T, F]
    dir_arr   = np.asarray(df['direction'], dtype=np.int64)
    tp_arr    = np.asarray(df['tp_pips'], dtype=np.float32)
    sl_arr    = np.asarray(df['sl_pips'], dtype=np.float32)

    n_total = len(df) - seq_len
    if n_total <= 0:
        return np.empty((0, seq_len, len(available)), dtype=np.float32), \
               np.empty(0, dtype=np.int64), \
               np.empty((0, 2), dtype=np.float32)

    X     = np.stack([feat_arr[i : i+seq_len] for i in range(n_total)])
    y_dir = dir_arr[seq_len:]
    y_mag = np.stack([tp_arr[seq_len:], sl_arr[seq_len:]], axis=1)
    return X, y_dir, y_mag


def save_shards(X, y_dir, y_mag, out_dir: str, pair: str, timeframe_min: int, split: str = "train"):
    """
    Save windowed samples as .npy files for fast DataLoader access.

    Output structure:
        data/forex/{pair}/{timeframe_min}/{split}/
            X.npy       — [N, seq_len, features]
            y_dir.npy   — [N] int64 direction classes
            y_mag.npy   — [N, 2] float32 (tp_pips, sl_pips)
    """
    shard_dir = os.path.join(out_dir, pair, str(timeframe_min), split)
    os.makedirs(shard_dir, exist_ok=True)

    np.save(os.path.join(shard_dir, "X.npy"),     X)
    np.save(os.path.join(shard_dir, "y_dir.npy"), y_dir)
    np.save(os.path.join(shard_dir, "y_mag.npy"), y_mag)
    print(f" [MT5Pipeline] Saved {len(X)} samples -> {shard_dir}")


def load_shard(shard_dir: str, mmap_mode: str | None = None) -> tuple:
    """
    Load a shard from disk. Returns (X, y_dir, y_mag) or empty arrays if not found.
    """
    X_path     = os.path.join(shard_dir, "X.npy")
    ydir_path  = os.path.join(shard_dir, "y_dir.npy")
    ymag_path  = os.path.join(shard_dir, "y_mag.npy")

    if not (os.path.exists(X_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return None, None, None

    X     = np.load(X_path,     mmap_mode=mmap_mode)
    y_dir = np.load(ydir_path,  mmap_mode=mmap_mode)
    y_mag = np.load(ymag_path,  mmap_mode=mmap_mode)
    return X, y_dir, y_mag


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_download_pipeline(
    pairs: list,
    timeframes: list,
    out_dir: str,
    n_bars: int = 50000,
    val_frac: float = 0.15,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    api_key: str | None = None,
):
    """
    Full end-to-end pipeline:
        1. Connect to MT5
        2. Download OHLCV bars
        3. Compute indicators
        4. Generate labels
        5. Normalize
        6. Build windows
        7. Split train/val
        8. Save shards

    Args:
        pairs:      List of currency pair symbols.
        timeframes: List of timeframe integers (minutes).
        out_dir:    Output directory for .npy shards.
        n_bars:     Number of bars to download per pair/timeframe.
        val_frac:   Fraction of data to use for validation.
        login:      MT5 demo account login ID.
        password:   MT5 demo account password.
        server:     MT5 demo account server name.
        api_key:    MT5 API key.
    """
    if not connect_mt5(login=login, password=password, server=server, api_key=api_key):
        print(" [MT5Pipeline] BLOCKED: Cannot proceed without MT5 connection.")
        print(" [MT5Pipeline] Provide demo account credentials via CLI (--login, --password, --server, --api_key) or env vars (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_API_KEY).")
        return

    try:
        for pair in pairs:
            for tf in timeframes:
                seq_len = TIMEFRAME_LOOKBACK.get(tf, 168)
                print(f"\n [MT5Pipeline] Processing {pair} @ {tf}min (lookback={seq_len})...")

                try:
                    df = download_bars(pair, tf, n_bars)
                    df = compute_indicators(df)
                    df = generate_labels(df, pair)
                    df = normalize_ohlcv(df)

                    # Train/val split (chronological — no random shuffling for time-series)
                    split_idx = int(len(df) * (1.0 - val_frac))
                    train_df  = df.iloc[:split_idx]
                    val_df    = df.iloc[split_idx:]

                    X_tr, yd_tr, ym_tr = build_windows(train_df, seq_len)
                    X_va, yd_va, ym_va = build_windows(val_df, seq_len)

                    save_shards(X_tr, yd_tr, ym_tr, out_dir, pair, tf, "train")
                    save_shards(X_va, yd_va, ym_va, out_dir, pair, tf, "val")

                except Exception as e:
                    print(f" [MT5Pipeline] ERROR for {pair}@{tf}min: {e}")
                    continue
    finally:
        disconnect_mt5()

    print("\n [MT5Pipeline] Download pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LemGendary MT5 Data Pipeline")
    parser.add_argument("--mode",        choices=["download", "build_dataset", "check"], default="check",
                        help="Pipeline mode: 'download' fetches live data, 'check' validates existing shards.")
    parser.add_argument("--pairs",       nargs="+", default=MAJOR_PAIRS,
                        help="Currency pairs to download (default: all majors)")
    parser.add_argument("--timeframes",  nargs="+", type=int, default=[60, 240],
                        help="Timeframes in minutes (default: H1 H4)")
    parser.add_argument("--out_dir",     default="data/forex",
                        help="Output directory for .npy shards")
    parser.add_argument("--n_bars",      type=int, default=50000,
                        help="Number of bars to download per pair/timeframe")
    parser.add_argument("--val_frac",    type=float, default=0.15,
                        help="Validation fraction (chronological split)")
    parser.add_argument("--login",       type=int, default=None,
                        help="MT5 demo account login ID (or set MT5_LOGIN env var)")
    parser.add_argument("--password",    type=str, default=None,
                        help="MT5 demo account password (or set MT5_PASSWORD env var)")
    parser.add_argument("--server",      type=str, default=None,
                        help="MT5 demo account server name (or set MT5_SERVER env var)")
    parser.add_argument("--api_key",     type=str, default=None,
                        help="MT5 demo account API key (or set MT5_API_KEY env var)")
    args = parser.parse_args()

    if args.mode == "check":
        print(" [MT5Pipeline] Checking existing shards in:", args.out_dir)
        for pair in args.pairs:
            for tf in args.timeframes:
                for split in ["train", "val"]:
                    shard_dir = os.path.join(args.out_dir, pair, str(tf), split)
                    X, y_dir, _ = load_shard(shard_dir)
                    if X is not None:
                        print(f"   [OK] {pair}/{tf}min/{split}: {len(X)} samples, shape {X.shape}")
                    else:
                        print(f"   [MISSING] {pair}/{tf}min/{split}: No shard found")

    elif args.mode == "download":
        print(" [MT5Pipeline] Starting download pipeline...")
        print(" [MT5Pipeline] NOTE: Requires MetaTrader 5 terminal running with demo account.")
        run_download_pipeline(
            pairs      = args.pairs,
            timeframes = args.timeframes,
            out_dir    = args.out_dir,
            n_bars     = args.n_bars,
            val_frac   = args.val_frac,
            login      = args.login,
            password   = args.password,
            server     = args.server,
            api_key    = args.api_key,
        )


if __name__ == "__main__":
    main()
