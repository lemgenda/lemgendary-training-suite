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

# pylint: disable=no-member,too-many-return-statements,duplicate-code
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirrors forex_predictor.py)
# Currency Universe: Titan 4 starting core, extensible up to 16 professional assets
TITAN_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
MAJOR_PAIRS = TITAN_PAIRS

EXTENDED_PAIRS = [
    # G7 Majors (4 Core + 3 G7)
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    # High-Beta Crosses
    "EURJPY", "GBPJPY", "EURGBP",
    # Commodities & Energy
    "XAGUSD", "USOIL",
    # Global Equity Indices
    "US500", "USTEC", "GER40"
]

# Currency pair index (0-based) - shared across suite
PAIR_INDEX = {p: i for i, p in enumerate(EXTENDED_PAIRS)}
NUM_PAIRS = len(PAIR_INDEX)

# Ordered list of timeframe rungs in minutes (Governor curriculum ladder)
TIMEFRAME_RUNGS = [1, 5, 15, 60, 240, 1440]

# Canonical lookback windows per timeframe (covers ~1 trading week each)
TIMEFRAME_LOOKBACK = {
    1:    512,   # M1  → ~8.5 hours
    5:    288,   # M5  → ~1 day
    15:   192,   # M15 → ~2 days
    60:   168,   # H1  → ~1 week
    240:   90,   # H4  → ~2.5 weeks
    1440:  252,  # D1  → ~1 year
}

# MetaTrader 5 Timeframe Attribute Mapping
MT5_TIMEFRAMES = {
    1: "TIMEFRAME_M1",
    5: "TIMEFRAME_M5",
    15: "TIMEFRAME_M15",
    60: "TIMEFRAME_H1",
    240: "TIMEFRAME_H4",
    1440: "TIMEFRAME_D1",
}

# ─────────────────────────────────────────────────────────────────────────────
# 6-Fold Anchored Walk-Forward Matrix (2019 – 2026) with 14-Day Embargo
# ─────────────────────────────────────────────────────────────────────────────
WALK_FORWARD_FOLDS = {
    "fold_1": {
        "train_start": "2019-01-01", "train_end": "2020-12-31",
        "regime": "Pre-Pandemic & Peak Volatility (2 Years)"
    },
    "fold_2": {
        "train_start": "2021-01-01", "train_end": "2021-12-31",
        "regime": "Recovery & Supply Chain Stress"
    },
    "fold_3": {
        "train_start": "2022-01-01", "train_end": "2022-12-31",
        "regime": "Global Rate Hikes & Dollar Surge"
    },
    "fold_4": {
        "train_start": "2023-01-01", "train_end": "2023-12-31",
        "regime": "Inflation Peaks & Consolidation"
    },
    "fold_5": {
        "train_start": "2024-01-01", "train_end": "2024-12-31",
        "regime": "Central Bank Pivot"
    },
    "fold_6": {
        "train_start": "2025-01-01", "train_end": "2025-12-31",
        "regime": "Modern High-Fidelity Consolidations"
    },
    "val": {
        "train_start": "2026-01-01", "train_end": "2026-12-31",
        "regime": "Global Validation Set"
    }
}

FEATURES = [
    "open", "high", "low", "close", "volume",     # OHLCV (5)
    "rsi", "macd", "macd_signal", "atr", "bb_width",  # Indicators (5)
    "session_sin", "session_cos",                  # Session encoding (2)
    "atr_percentile", "bar_range_ratio",           # Volatility regime (2)
]  # Total: 14 features

# Label generation: forward-looking window to measure price movement
LABEL_HORIZON_BARS = 20       # Look N bars ahead for TP/SL determination
DIRECTION_THRESHOLD_PIPS = 5  # 5-pip threshold produces healthy 3-class distribution (Down/Sideways/Up)

# Per-pair average spread in pips (deducted at label generation time for realistic trade economics)
PAIR_SPREADS_PIPS = {
    "EURUSD": 1.2,  "GBPUSD": 1.5,  "USDJPY": 1.3,  "XAUUSD": 35.0,
    "USDCAD": 1.8,  "USDCHF": 1.5,  "AUDUSD": 1.4,  "NZDUSD": 2.0,
    "EURGBP": 1.5,  "EURJPY": 1.8,  "GBPJPY": 2.5,  "USOIL":  40.0,
    "US500":  10.0, "USTEC":  20.0,  "GER40":  10.0,  "XAGUSD": 20.0,
}

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

    # Fast non-blocking check: Only attempt IPC attach if MT5 terminal is running or explicitly enabled
    if os.environ.get("MT5_ENABLE", "0") != "1":
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, "MetaTrader 5")
                if not hwnd:
                    print(" [MT5] MetaTrader 5 terminal window not detected. Operating in offline/mock mode.")
                    return False
            except Exception:
                return False
        else:
            return False

    # First try attaching directly to the running MT5 terminal session
    try:
        if mt5.initialize():  # type: ignore
            info = mt5.account_info()  # type: ignore
            if info:
                print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
                return True
    except Exception:
        pass

    # Fallback to explicit login parameters if active session not found
    init_kwargs = {}
    if login and password and server:
        init_kwargs.update({"login": login, "password": password, "server": server})

    try:
        if not mt5.initialize(**init_kwargs):  # type: ignore
            err = mt5.last_error()  # type: ignore
            print(f" [MT5] Initialize failed: {err}")
            return False
    except Exception as e:
        print(f" [MT5] Initialize exception: {e}")
        return False

    info = mt5.account_info()  # type: ignore
    if info:
        print(f" [MT5] Connected -> Account: {info.login} | Server: {info.server} | Balance: {info.balance} {info.currency}")
        return True

    print(" [MT5] Connected to MT5 terminal session.")
    return True


def disconnect_mt5():
    """Cleanly shut down the MT5 connection."""
    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
        mt5.shutdown()  # type: ignore
        print(" [MT5] Disconnected.")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Data Download
# ─────────────────────────────────────────────────────────────────────────────

def generate_mock_bars(pair: str, timeframe_min: int, n_bars: int | None = None, start_date: str = "2019-01-01") -> pd.DataFrame:
    """Generate realistic synthetic OHLCV bars spanning 2019-01-01 to present for all 16 assets."""
    np.random.seed(abs(hash(pair + str(timeframe_min))) % (2**32 - 1))
    
    price_map = {
        "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 155.00, "XAUUSD": 2400.00,
        "USDCAD": 1.3650, "USDCHF": 0.8950, "AUDUSD": 0.6650, "NZDUSD": 0.6100,
        "EURJPY": 168.00, "GBPJPY": 196.50, "EURGBP": 0.8550,
        "XAGUSD": 29.50,  "USOIL": 78.50,
        "US500":  5500.0, "USTEC": 19500.0, "GER40": 18500.0
    }
    base_price = price_map.get(pair, 1.0000)
    
    if any(x in pair for x in ["JPY", "XAG"]):
        pip_size = 0.01
    elif any(x in pair for x in ["XAU", "USOIL"]):
        pip_size = 0.1
    elif any(x in pair for x in ["US500", "USTEC", "GER40"]):
        pip_size = 1.0
    else:
        pip_size = 0.0001

    dt_end = datetime.now(timezone.utc)
    freq = f"{timeframe_min}min" if timeframe_min < 1440 else "1D"
    
    if n_bars is not None and n_bars > 0 and start_date is None:
        times = pd.date_range(end=dt_end, periods=n_bars, freq=freq)
    else:
        start_dt = pd.to_datetime(start_date, utc=True) if start_date else pd.to_datetime("2019-01-01", utc=True)
        times = pd.date_range(start=start_dt, end=dt_end, freq=freq)
        if n_bars is not None and len(times) < n_bars:
            times = pd.date_range(end=dt_end, periods=n_bars, freq=freq)

    total_bars = len(times)
    returns = np.random.normal(loc=0.0, scale=0.001, size=total_bars)
    price_curve = base_price * np.exp(np.cumsum(returns))
    
    spread = pip_size * np.random.uniform(1.0, 3.0, size=total_bars)
    highs = price_curve + np.abs(np.random.normal(0, pip_size * 5, size=total_bars)) + spread
    lows = price_curve - np.abs(np.random.normal(0, pip_size * 5, size=total_bars)) - spread
    opens = price_curve + np.random.uniform(-spread, spread, size=total_bars)
    closes = price_curve
    
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    volumes = np.random.randint(100, 5000, size=total_bars)
    
    print(f" [MT5Pipeline] Synthesized {total_bars} bars for {pair} {timeframe_min}min (Spans {times[0].strftime('%Y-%m-%d')} -> {times[-1].strftime('%Y-%m-%d')})")
    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })


def download_bars(pair: str, timeframe_min: int, n_bars: int = 50000, start_date: str = "2019-01-01") -> pd.DataFrame:
    """
    Download historical OHLCV bars from MT5 for a given pair and timeframe.

    Args:
        pair:          Symbol (e.g. "EURUSD")
        timeframe_min: Timeframe in minutes (1, 5, 15, 60, 240, 1440)
        n_bars:        Number of bars to download
        start_date:    Start date string (e.g. "2019-01-01")

    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import-untyped]
        tf_attr = MT5_TIMEFRAMES.get(timeframe_min)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_min}min")

        tf = getattr(mt5, tf_attr)
        rates = None
        if start_date:
            try:
                dt_from = pd.to_datetime(start_date, utc=True).to_pydatetime()
                dt_to = datetime.now(timezone.utc)
                rates = mt5.copy_rates_range(pair, tf, dt_from, dt_to)  # type: ignore
            except Exception:
                rates = None

        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_from_pos(pair, tf, 0, n_bars)  # type: ignore

        if rates is None or len(rates) == 0:
            error = mt5.last_error()  # type: ignore
            raise RuntimeError(f"No data returned for {pair} {timeframe_min}min: {error}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['volume'] = df['tick_volume']
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        print(f" [MT5] Downloaded {len(df)} bars for {pair} {timeframe_min}min (Spans {pd.Series(df['time']).iloc[0].strftime('%Y-%m-%d')} -> {pd.Series(df['time']).iloc[-1].strftime('%Y-%m-%d')})")
        assert isinstance(df, pd.DataFrame)
        return df

    except ImportError as exc:
        raise RuntimeError(
            "[MT5] MetaTrader5 not installed. Run: pip install MetaTrader5\n"
            "      Ensure MT5 terminal is running with a demo account."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators (pandas-ta backed, no MT5 dependency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and regime features to OHLCV DataFrame.
    Uses pandas-ta if available and compatible, falls back to manual NumPy implementations.

    Indicators added (10 -> 14 total features):
        rsi          -- RSI(14)
        macd         -- MACD line (12-26-9), normalized by close
        macd_signal  -- MACD signal line, normalized by close
        atr          -- ATR(14), normalized by close price
        bb_width     -- Bollinger Band width (20, 2sigma), normalized by close
        session_sin  -- sine encoding of hour-of-day for cyclical time representation
        session_cos  -- cosine encoding of hour-of-day
        atr_percentile -- ATR rank over rolling 100-bar window, normalized [0, 1]
        bar_range_ratio -- (high - low) / (atr + 1e-8), intrabar volatility vs. trend

    Args:
        df: DataFrame with [time, open, high, low, close, volume] columns.

    Returns:
        DataFrame with 9 additional indicator columns (14 features total).
    """
    df = df.copy()
    close = np.asarray(df['close'], dtype=np.float64)
    high  = np.asarray(df['high'],  dtype=np.float64)
    low   = np.asarray(df['low'],   dtype=np.float64)

    computed_ta = False
    try:
        import pandas_ta as ta  # type: ignore
        df_ta = df.copy()
        df_ta.ta.rsi(length=14, append=True)
        df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_ta.ta.atr(length=14, append=True)
        df_ta.ta.bbands(length=20, std=2, append=True)

        rsi_col    = next((c for c in df_ta.columns if c.upper().startswith('RSI_')), None)
        macd_col   = next((c for c in df_ta.columns if 'MACD_' in c.upper() and 'H' not in c.upper() and 'S' not in c.upper()), None)
        macds_col  = next((c for c in df_ta.columns if 'MACDS_' in c.upper()), None)
        atr_col    = next((c for c in df_ta.columns if c.upper().startswith('ATRR_') or c.upper().startswith('ATR')), None)
        bbu_col    = next((c for c in df_ta.columns if 'BBU_' in c.upper()), None)
        bbl_col    = next((c for c in df_ta.columns if 'BBL_' in c.upper()), None)

        if rsi_col and macd_col and macds_col and atr_col and bbu_col and bbl_col:
            df['rsi']         = (np.asarray(df_ta[rsi_col], dtype=np.float64) / 100.0).tolist()
            df['macd']        = (np.asarray(df_ta[macd_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['macd_signal'] = (np.asarray(df_ta[macds_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['atr']         = (np.asarray(df_ta[atr_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df['bb_width']    = ((np.asarray(df_ta[bbu_col], dtype=np.float64) - np.asarray(df_ta[bbl_col], dtype=np.float64)) / (close + 1e-8)).tolist()
            computed_ta = True
    except Exception:
        computed_ta = False

    if not computed_ta:
        df['rsi']         = _rsi_manual(close, 14).tolist()
        macd_line, sig    = _macd_manual(close, 12, 26, 9)
        df['macd']        = (macd_line / (close + 1e-8)).tolist()
        df['macd_signal'] = (sig       / (close + 1e-8)).tolist()
        df['atr']         = (_atr_manual(high, low, close, 14) / (close + 1e-8)).tolist()
        df['bb_width']    = (_bbwidth_manual(close, 20, 2)     / (close + 1e-8)).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    # Session encoding: cyclical hour-of-day (works for both datetime index and 'time' column)
    if 'time' in df.columns:
        hours = pd.to_datetime(df['time'], utc=True).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour  # type: ignore
    else:
        hours = pd.Series([0.0] * len(df))
    hour_norm = np.asarray(hours, dtype=np.float64) / 23.0 * 2.0 * np.pi
    df['session_sin'] = np.sin(hour_norm).tolist()
    df['session_cos'] = np.cos(hour_norm).tolist()

    # Volatility regime features using the already-computed raw ATR
    raw_atr = _atr_manual(np.asarray(df['high'], dtype=np.float64),
                          np.asarray(df['low'],  dtype=np.float64),
                          np.asarray(df['close'], dtype=np.float64), 14)
    # ATR percentile: rolling 100-bar rank normalized to [0, 1]
    atr_series = pd.Series(list(raw_atr), dtype=float)
    atr_pct = atr_series.rolling(100, min_periods=1).apply(
        lambda x: float(np.sum(x <= x[-1])) / float(len(x)), raw=True
    )
    df['atr_percentile'] = np.asarray(atr_pct, dtype=np.float32).tolist()

    # Bar range ratio: (high - low) / (ATR + 1e-8) — intrabar volatility relative to trend
    high_arr = np.asarray(df['high'], dtype=np.float64)
    low_arr  = np.asarray(df['low'],  dtype=np.float64)
    df['bar_range_ratio'] = ((high_arr - low_arr) / (raw_atr + 1e-8)).astype(np.float32).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
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
    Generate direction + magnitude labels for each bar using the Triple Barrier Method.

    Direction (3-class):
        0 = Down     (net downward move > threshold_pips and dominates within horizon)
        1 = Sideways (move < threshold_pips in both directions)
        2 = Up       (net upward move > threshold_pips and dominates within horizon)

    Magnitude (spread-adjusted for realistic trade economics):
        tp_pips = peak upward move minus per-pair average spread
        sl_pips = peak downward move plus per-pair average spread

    Pip size per pair:
        JPY/XAU pairs: 0.01 pip size
        US500/USTEC/GER40/USOIL: index-specific pip sizes
        Others: 0.0001 pip size
    """
    if any(x in pair for x in ["JPY"]):
        pip_size = 0.01
    elif "XAU" in pair:
        pip_size = 0.1
    elif any(x in pair for x in ["US500", "USTEC", "GER40"]):
        pip_size = 1.0
    elif "USOIL" in pair or "XAG" in pair:
        pip_size = 0.01
    else:
        pip_size = 0.0001

    # Per-pair average spread in pips for realistic label economics
    spread_pips = PAIR_SPREADS_PIPS.get(pair, 2.0)

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

        # Spread-adjusted magnitudes: TP shrinks (spread cost), SL grows (spread widens stop)
        tp_pips[i] = max(0.0, float(up_move) - spread_pips)
        sl_pips[i] = max(0.0, float(down_move) + spread_pips)

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

def build_windows(df: pd.DataFrame, seq_len: int, max_samples: int = 50000, stride: int | None = None) -> tuple:
    """
    Slice the processed DataFrame into (features, direction, magnitude, timestamps) windows.
    Uses memory-efficient preallocation and uniform strided sampling for ultra-dense timeframes (M1/M5)
    to prevent multi-gigabyte RAM overflows while maintaining 100% historical multi-regime coverage.

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
            timestamps: [N]                        int64 Unix timestamps (seconds) of each window's last bar
    """
    available = [c for c in FEATURES if c in df.columns]
    feat_arr  = np.asarray(df[available], dtype=np.float32)    # [T, F]
    dir_arr   = np.asarray(df['direction'], dtype=np.int64)
    tp_arr    = np.asarray(df['tp_pips'], dtype=np.float32)
    sl_arr    = np.asarray(df['sl_pips'], dtype=np.float32)

    # Build Unix timestamp array for precise multi-timeframe alignment
    if 'time' in df.columns:
        ts_arr = np.asarray(
            pd.to_datetime(df['time'], utc=True).astype(np.int64) // 10**9,
            dtype=np.int64
        )
    else:
        ts_arr = np.arange(len(df), dtype=np.int64)

    n_total = len(df) - seq_len
    if n_total <= 0:
        return (np.empty((0, seq_len, len(available)), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64))

    if stride is None:
        stride = max(1, int(np.ceil(n_total / max_samples))) if max_samples > 0 else 1

    indices = np.arange(0, n_total, stride)
    num_samples = len(indices)

    X = np.empty((num_samples, seq_len, len(available)), dtype=np.float32)
    for out_i, i in enumerate(indices):
        X[out_i] = feat_arr[i : i + seq_len]

    target_indices = indices + seq_len
    y_dir      = dir_arr[target_indices]
    y_mag      = np.stack([tp_arr[target_indices], sl_arr[target_indices]], axis=1)
    timestamps = ts_arr[target_indices]
    return X, y_dir, y_mag, timestamps


def save_shards(X, y_dir, y_mag, out_dir: str, pair: str, timeframe_min: int, split: str = "train", timestamps=None):
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

    np.save(os.path.join(shard_dir, "X.npy"),     X)
    np.save(os.path.join(shard_dir, "y_dir.npy"), y_dir)
    np.save(os.path.join(shard_dir, "y_mag.npy"), y_mag)
    if timestamps is not None:
        np.save(os.path.join(shard_dir, "timestamps.npy"), timestamps)
    print(f" [MT5Pipeline] Saved {len(X)} samples -> {shard_dir}")


def load_shard(shard_dir: str, mmap_mode: Literal['c', 'r', 'r+', 'w+'] | None = None) -> tuple:
    """
    Load a shard from disk. Returns (X, y_dir, y_mag, timestamps) or (None, None, None, None) if not found.
    timestamps is optional — may be None if shard was built before v2.0.
    """
    X_path     = os.path.join(shard_dir, "X.npy")
    ydir_path  = os.path.join(shard_dir, "y_dir.npy")
    ymag_path  = os.path.join(shard_dir, "y_mag.npy")
    ts_path    = os.path.join(shard_dir, "timestamps.npy")

    if not (os.path.exists(X_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return None, None, None, None

    X          = np.load(X_path,    mmap_mode=mmap_mode)  # type: ignore
    y_dir      = np.load(ydir_path, mmap_mode=mmap_mode)  # type: ignore
    y_mag      = np.load(ymag_path, mmap_mode=mmap_mode)  # type: ignore
    timestamps = np.load(ts_path,   mmap_mode=mmap_mode) if os.path.exists(ts_path) else None  # type: ignore
    return X, y_dir, y_mag, timestamps


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
            t_end   = pd.to_datetime(fold_info["train_end"],   utc=True)
            mask    = (df_time >= t_start) & (df_time <= t_end)
            fold_df = df[mask]
        else:
            # Chronological fallback (dummy logic for synthetic without time)
            n_total = len(df)
            start_idx = int(n_total * (list(WALK_FORWARD_FOLDS.keys()).index(fold_id) * 0.1))
            end_idx = min(n_total, start_idx + int(n_total * 0.1))
            fold_df = df.iloc[start_idx:end_idx]
        assert isinstance(fold_df, pd.DataFrame)

        if len(fold_df) > seq_len:
            X, yd, ym, ts = build_windows(fold_df, seq_len)
            save_shards(X, yd, ym, out_dir, pair, tf, split=f"folds/{fold_id}", timestamps=ts)
            print(f"   [{fold_id.upper()}] Generated {pair}@{tf}min ({len(X)} samples) -> {fold_info['regime']}")


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_download_pipeline(
    pairs: list,
    timeframes: list,
    out_dir: str,
    n_bars: int = 50000,
    start_date: str = "2019-01-01",
    val_frac: float = 0.15,
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
        print(" [MT5Pipeline] MT5 connection inactive. Switching to Synthetic OHLCV Generator Mode.")

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

                except Exception as e:
                    print(f" [MT5Pipeline] ERROR for {pair}@{tf}min: {e}")
                    import traceback
                    traceback.print_exc()
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
    parser.add_argument("--pairs",       nargs="+", default=EXTENDED_PAIRS,
                        help="Currency pairs / instruments to include (default: all 16 extended instruments)")
    parser.add_argument("--timeframes",  nargs="+", type=int, default=[1, 5, 15, 60, 240, 1440],
                        help="Timeframes in minutes (default: M1, M5, M15, H1, H4, D1)")
    parser.add_argument("--out_dir",     default="data/forex",
                        help="Output directory for .npy shards")
    parser.add_argument("--n_bars",      type=int, default=50000,
                        help="Number of bars to download per pair/timeframe")
    parser.add_argument("--start_date",  type=str, default="2019-01-01",
                        help="Start date for historical coverage (default: 2019-01-01)")
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
                    X, y_dir, y_mag = load_shard(shard_dir, mmap_mode="r")
                    if X is not None:
                        import numpy as np
                        corrupt = False
                        for tensor, name in [(X, 'X'), (y_dir, 'y_dir'), (y_mag, 'y_mag')]:
                            if tensor is not None and (np.isnan(tensor).any() or np.isinf(tensor).any()):
                                print(f"   [CORRUPT] {pair}/{tf}min/{split}: Tensor {name} contains NaN or Inf values!")
                                corrupt = True
                        if not corrupt:
                            print(f"   [OK] {pair}/{tf}min/{split}: {len(X)} samples pristine, shape {X.shape}")
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
            start_date = args.start_date,
            val_frac   = args.val_frac,
            login      = args.login,
            password   = args.password,
            server     = args.server,
            api_key    = args.api_key,
        )


if __name__ == "__main__":
    main()
