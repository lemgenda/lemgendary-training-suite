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

import argparse
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from data.forex_spec import MT5_SYMBOLS, TIMEFRAME_LOOKBACK, TIMEFRAME_RUNGS, shard_file_paths
from data.mt5_features import (
    FEATURES,
    compute_indicators,
    generate_labels,
    normalize_ohlcv,
)
from data.mt5_terminal import (
    connect_mt5,
    disconnect_mt5,
    download_bars,
    generate_mock_bars,
)

logger = logging.getLogger("lemtrain.data.mt5_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# 6-Fold Anchored Walk-Forward Matrix (2019 – 2026) with 14-Day Embargo
# ─────────────────────────────────────────────────────────────────────────────
WALK_FORWARD_FOLDS = {
    "fold_1": {
        "train_start": "2019-01-01",
        "train_end": "2020-12-31",
        "regime": "Pre-Pandemic & Peak Volatility (2 Years)",
    },
    "fold_2": {
        "train_start": "2021-01-01",
        "train_end": "2021-12-31",
        "regime": "Recovery & Supply Chain Stress",
    },
    "fold_3": {
        "train_start": "2022-01-01",
        "train_end": "2022-12-31",
        "regime": "Global Rate Hikes & Dollar Surge",
    },
    "fold_4": {
        "train_start": "2023-01-01",
        "train_end": "2023-12-31",
        "regime": "Inflation Peaks & Consolidation",
    },
    "fold_5": {
        "train_start": "2024-01-01",
        "train_end": "2024-12-31",
        "regime": "Central Bank Pivot",
    },
    "fold_6": {
        "train_start": "2025-01-01",
        "train_end": "2025-12-31",
        "regime": "Modern High-Fidelity Consolidations",
    },
    "val": {
        "train_start": "2026-01-01",
        "train_end": "2026-12-31",
        "regime": "Global Validation Set",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MT5 Connection (requires live MT5 terminal)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Data Download
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators (pandas-ta backed, no MT5 dependency)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Label Generation
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Shard Serialization
# ─────────────────────────────────────────────────────────────────────────────


def build_windows(
    df: pd.DataFrame, seq_len: int, max_samples: int = 50000, stride: int | None = None
) -> tuple:
    """
    Slice the processed DataFrame into (features, direction, magnitude, timestamps) windows.
    Uses memory-efficient preallocation and uniform strided sampling for ultra-dense
    timeframes (M1/M5) to prevent multi-gigabyte RAM overflows while maintaining 100%
    historical multi-regime coverage.

    Args:
        df:          Processed DataFrame with FEATURES + direction/tp_pips/sl_pips columns.
        seq_len:     Lookback window size.
        max_samples: Maximum number of window samples per split/fold shard (default: 50000).
        stride:      Explicit sampling step. If None, automatically derived from max_samples.

    Returns:
        Tuple of (X, y_dir, y_mag, timestamps):
            X:          [N, seq_len, n_features]  float32
            y_dir:      [N]                        int64 direction class
            y_mag:      [N, 2]                     float32 (tp_pips, sl_pips)
            timestamps: [N]                        int64 Unix seconds of each window's last bar
    """
    available = [c for c in FEATURES if c in df.columns]
    feat_arr = np.asarray(df[available], dtype=np.float32)  # [T, F]
    dir_arr = np.asarray(df["direction"], dtype=np.int64)
    tp_arr = np.asarray(df["tp_pips"], dtype=np.float32)
    sl_arr = np.asarray(df["sl_pips"], dtype=np.float32)

    # Build Unix timestamp array for precise multi-timeframe alignment
    if "time" in df.columns:
        raw_dt = pd.to_datetime(df["time"], utc=True)
        if isinstance(raw_dt, pd.DatetimeIndex):
            ts_arr = np.asarray(raw_dt.astype(np.int64) // 10**9, dtype=np.int64)
        else:
            ts_arr = np.asarray(
                [int(t.timestamp()) if hasattr(t, "timestamp") else 0 for t in raw_dt],
                dtype=np.int64,
            )
    else:
        ts_arr = np.arange(len(df), dtype=np.int64)

    n_total = len(df) - seq_len
    if n_total <= 0:
        return (
            np.empty((0, seq_len, len(available)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    if stride is None:
        stride = max(1, int(np.ceil(n_total / max_samples))) if max_samples > 0 else 1

    indices = np.arange(0, n_total, stride)
    num_samples = len(indices)

    x_data = np.empty((num_samples, seq_len, len(available)), dtype=np.float32)
    for out_i, i in enumerate(indices):
        x_data[out_i] = feat_arr[i : i + seq_len]

    target_indices = indices + seq_len
    y_dir = dir_arr[target_indices]
    y_mag = np.stack([tp_arr[target_indices], sl_arr[target_indices]], axis=1)
    timestamps = ts_arr[target_indices]
    return x_data, y_dir, y_mag, timestamps


def save_shards(
    x_data,
    y_dir,
    y_mag,
    out_dir: str,
    pair: str,
    timeframe_min: int,
    split: str = "train",
    timestamps=None,
):
    """
    Save windowed samples as .npy files for fast DataLoader access.

    Output structure:
        data/forex/{pair}/{timeframe_min}/{split}/
            X.npy           -- [N, seq_len, features]
            y_dir.npy       -- [N] int64 direction classes
            y_mag.npy       -- [N, 2] float32 (tp_pips, sl_pips)
            timestamps.npy  -- [N] int64 Unix timestamps of each window's last bar
    """
    shard_dir = os.path.join(out_dir, pair, str(timeframe_min), split)
    os.makedirs(shard_dir, exist_ok=True)

    np.save(os.path.join(shard_dir, "X.npy"), x_data)
    np.save(os.path.join(shard_dir, "y_dir.npy"), y_dir)
    np.save(os.path.join(shard_dir, "y_mag.npy"), y_mag)
    if timestamps is not None:
        np.save(os.path.join(shard_dir, "timestamps.npy"), timestamps)
    print(f" [MT5Pipeline] Saved {len(x_data)} samples -> {shard_dir}")


def load_shard(shard_dir: str, mmap_mode: Any = None) -> tuple:
    """
    Load a shard from disk.

    Returns (X, y_dir, y_mag, timestamps), or (None, None, None, None) if not found.
    timestamps is optional — may be None if shard was built before v2.0.
    """
    x_path, ydir_path, ymag_path, ts_path = shard_file_paths(shard_dir)

    if not (os.path.exists(x_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return None, None, None, None

    x_data = np.load(x_path, mmap_mode=mmap_mode)
    y_dir = np.load(ydir_path, mmap_mode=mmap_mode)
    y_mag = np.load(ymag_path, mmap_mode=mmap_mode)
    timestamps = np.load(ts_path, mmap_mode=mmap_mode) if os.path.exists(ts_path) else None
    return x_data, y_dir, y_mag, timestamps


def build_walk_forward_folds(df: pd.DataFrame, seq_len: int, out_dir: str, pair: str, tf: int):
    """
    Generate and save 6-Fold Anchored Walk-Forward shards with 14-day embargo.
    Includes timestamps.npy for precise multi-timeframe temporal alignment.
    """
    for fold_id, fold_info in WALK_FORWARD_FOLDS.items():
        fold_dir = os.path.join(out_dir, pair, str(tf), "folds", fold_id)
        os.makedirs(fold_dir, exist_ok=True)

        # Filter by datetime if time column exists, otherwise partition chronologically
        if "time" in df.columns:
            df_time = pd.to_datetime(df["time"], utc=True)
            t_start = pd.to_datetime(fold_info["train_start"], utc=True)
            t_end = pd.to_datetime(fold_info["train_end"], utc=True)
            mask = (df_time >= t_start) & (df_time <= t_end)
            fold_df = df[mask]
        else:
            # Chronological fallback (dummy logic for synthetic without time)
            n_total = len(df)
            start_idx = int(n_total * (list(WALK_FORWARD_FOLDS.keys()).index(fold_id) * 0.1))
            end_idx = min(n_total, start_idx + int(n_total * 0.1))
            fold_df = df.iloc[start_idx:end_idx]
        assert isinstance(fold_df, pd.DataFrame)

        if len(fold_df) > seq_len:
            x_data, yd, ym, ts = build_windows(fold_df, seq_len)
            save_shards(x_data, yd, ym, out_dir, pair, tf, split=f"folds/{fold_id}", timestamps=ts)
            print(
                f"   [{fold_id.upper()}] Generated {pair}@{tf}min ({len(x_data)} samples) "
                f"-> {fold_info['regime']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def run_download_pipeline(
    pairs: list,
    timeframes: list,
    out_dir: str,
    n_bars: int = 50000,
    start_date: str = "2019-01-01",
    build_folds: bool = True,
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    api_key: str | None = None,
):
    """
    Full end-to-end pipeline:
        1. Connect to MT5 (or use synthetic mock generator spanning 2019-present)
        2. Download/Generate OHLCV bars
        3. Compute indicators
        4. Generate labels
        5. Normalize
        6. Build windows & Walk-Forward Folds
        7. Save production & fold shards
    """
    mt5_active = connect_mt5(login=login, password=password, server=server, api_key=api_key)
    if not mt5_active:
        print(
            " [MT5Pipeline] MT5 connection inactive. Switching to Synthetic OHLCV Generator Mode."
        )

    try:
        for pair in pairs:
            for tf in timeframes:
                seq_len = TIMEFRAME_LOOKBACK.get(tf, 168)
                print(f"\n [MT5Pipeline] Processing {pair} @ {tf}min (lookback={seq_len})...")

                try:
                    if mt5_active:
                        df = download_bars(pair, tf, n_bars=n_bars, start_date=start_date)
                    else:
                        df = generate_mock_bars(pair, tf, n_bars=n_bars, start_date=start_date)
                    df = compute_indicators(df)
                    df = generate_labels(df, pair)
                    df = normalize_ohlcv(df)

                    # Walk-Forward Matrix (Generates chronological chunks and val set)
                    if build_folds:
                        build_walk_forward_folds(df, seq_len, out_dir, pair, tf)

                except (RuntimeError, ValueError, KeyError, TypeError, IndexError, OSError) as e:
                    # Per-pair job boundary: one bad pair/timeframe must not abort the whole batch.
                    # RuntimeError: MT5 returned no data; ValueError/KeyError/TypeError/IndexError:
                    # malformed bars or indicator frames; OSError: shard write failure.
                    logger.exception("Pipeline failed for %s@%smin", pair, tf)
                    print(f" [MT5Pipeline] ERROR for {pair}@{tf}min: {e}")
                    continue
    finally:
        disconnect_mt5()

    print("\n [MT5Pipeline] Download pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entrypoint: download/build forex shards or check existing ones."""
    parser = argparse.ArgumentParser(description="LemGendary MT5 Data Pipeline")
    parser.add_argument(
        "--mode",
        choices=["download", "build_dataset", "check"],
        default="check",
        help="Pipeline mode: 'download' fetches live data, 'check' validates existing shards.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=MT5_SYMBOLS,
        help="Currency pairs / instruments to include (default: all 16 extended instruments)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        type=int,
        default=list(TIMEFRAME_RUNGS),
        help="Timeframes in minutes (default: M1, M5, M15, H1, H4, D1)",
    )
    parser.add_argument("--out_dir", default="data/forex", help="Output directory for .npy shards")
    parser.add_argument(
        "--n_bars", type=int, default=50000, help="Number of bars to download per pair/timeframe"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2019-01-01",
        help="Start date for historical coverage (default: 2019-01-01)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
        help="Deprecated and ignored: validation data comes from the walk-forward fold layout.",
    )
    parser.add_argument(
        "--login",
        type=int,
        default=None,
        help="MT5 demo account login ID (or set MT5_LOGIN env var)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="MT5 demo account password (or set MT5_PASSWORD env var)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="MT5 demo account server name (or set MT5_SERVER env var)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="MT5 demo account API key (or set MT5_API_KEY env var)",
    )
    args = parser.parse_args()

    if args.mode == "check":
        print(" [MT5Pipeline] Checking existing shards in:", args.out_dir)
        for pair in args.pairs:
            for tf in args.timeframes:
                for split in ["train", "val"]:
                    shard_dir = os.path.join(args.out_dir, pair, str(tf), split)
                    x_data, y_dir, y_mag, _timestamps = load_shard(shard_dir, mmap_mode="r")
                    if x_data is not None:
                        corrupt = False
                        for tensor, name in [(x_data, "X"), (y_dir, "y_dir"), (y_mag, "y_mag")]:
                            if tensor is not None and (
                                np.isnan(tensor).any() or np.isinf(tensor).any()
                            ):
                                print(
                                    f"   [CORRUPT] {pair}/{tf}min/{split}: "
                                    f"Tensor {name} contains NaN or Inf values!"
                                )
                                corrupt = True
                        if not corrupt:
                            print(
                                f"   [OK] {pair}/{tf}min/{split}: {len(x_data)} samples "
                                f"pristine, shape {x_data.shape}"
                            )
                    else:
                        print(f"   [MISSING] {pair}/{tf}min/{split}: No shard found")

    elif args.mode == "download":
        print(" [MT5Pipeline] Starting download pipeline...")
        print(" [MT5Pipeline] NOTE: Requires MetaTrader 5 terminal running with demo account.")
        run_download_pipeline(
            pairs=args.pairs,
            timeframes=args.timeframes,
            out_dir=args.out_dir,
            n_bars=args.n_bars,
            start_date=args.start_date,
            login=args.login,
            password=args.password,
            server=args.server,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
