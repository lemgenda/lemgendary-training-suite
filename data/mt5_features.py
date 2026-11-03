"""Per-bar feature engineering and labelling for the MT5 forex pipeline.

Indicators, session and volatility-regime features, OHLCV normalization and TP/SL
direction labels with per-pair spread costs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from training.utils.lazy_imports import load_pandas_ta

logger = logging.getLogger("lemtrain.data.mt5_pipeline")


# OHLCV (5) + indicators (5): the per-bar columns the live signal path also feeds the model.
PRICE_INDICATOR_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "atr",
    "bb_width",
]

FEATURES = [
    *PRICE_INDICATOR_FEATURES,
    "session_sin",
    "session_cos",  # Session encoding (2)
    "atr_percentile",
    "bar_range_ratio",  # Volatility regime (2)
]  # Total: 14 features

# Label generation: forward-looking window to measure price movement
LABEL_HORIZON_BARS = 20  # Look N bars ahead for TP/SL determination

DIRECTION_THRESHOLD_PIPS = (
    5  # 5-pip threshold produces healthy 3-class distribution (Down/Sideways/Up)
)

# Per-pair average spread in pips (deducted at label generation time for realistic trade economics)
PAIR_SPREADS_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.3,
    "XAUUSD": 35.0,
    "USDCAD": 1.8,
    "USDCHF": 1.5,
    "AUDUSD": 1.4,
    "NZDUSD": 2.0,
    "EURGBP": 1.5,
    "EURJPY": 1.8,
    "GBPJPY": 2.5,
    "USOIL": 40.0,
    "US500": 10.0,
    "USTEC": 20.0,
    "GER40": 10.0,
    "XAGUSD": 20.0,
}


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
    close = np.asarray(df["close"], dtype=np.float64)
    high = np.asarray(df["high"], dtype=np.float64)
    low = np.asarray(df["low"], dtype=np.float64)

    computed_ta = False
    try:
        ta = load_pandas_ta()

        # Each indicator returns a Series/DataFrame (or None when there is too little data);
        # pd.concat drops the None entries, matching the accessor's ``append=True`` behaviour.
        df_ta = pd.concat(
            [
                df,
                ta.rsi(df["close"], length=14),
                ta.macd(df["close"], fast=12, slow=26, signal=9),
                ta.atr(df["high"], df["low"], df["close"], length=14),
                ta.bbands(df["close"], length=20, std=2),
            ],
            axis=1,
        )

        rsi_col = next((c for c in df_ta.columns if c.upper().startswith("RSI_")), None)
        macd_col = next(
            (
                c
                for c in df_ta.columns
                if "MACD_" in c.upper() and "H" not in c.upper() and "S" not in c.upper()
            ),
            None,
        )
        macds_col = next((c for c in df_ta.columns if "MACDS_" in c.upper()), None)
        atr_col = next(
            (
                c
                for c in df_ta.columns
                if c.upper().startswith("ATRR_") or c.upper().startswith("ATR")
            ),
            None,
        )
        bbu_col = next((c for c in df_ta.columns if "BBU_" in c.upper()), None)
        bbl_col = next((c for c in df_ta.columns if "BBL_" in c.upper()), None)

        if rsi_col and macd_col and macds_col and atr_col and bbu_col and bbl_col:
            df["rsi"] = (np.asarray(df_ta[rsi_col], dtype=np.float64) / 100.0).tolist()
            df["macd"] = (np.asarray(df_ta[macd_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df["macd_signal"] = (
                np.asarray(df_ta[macds_col], dtype=np.float64) / (close + 1e-8)
            ).tolist()
            df["atr"] = (np.asarray(df_ta[atr_col], dtype=np.float64) / (close + 1e-8)).tolist()
            df["bb_width"] = (
                (
                    np.asarray(df_ta[bbu_col], dtype=np.float64)
                    - np.asarray(df_ta[bbl_col], dtype=np.float64)
                )
                / (close + 1e-8)
            ).tolist()
            computed_ta = True
    except ImportError as exc:
        logger.debug("pandas_ta_classic not installed; using manual indicators: %s", exc)
        computed_ta = False
    except (AttributeError, ValueError, KeyError, TypeError) as exc:
        # pandas_ta accessor missing or incompatible with the installed pandas/numpy.
        logger.warning(
            "pandas_ta_classic indicator computation failed; using manual indicators: %s", exc
        )
        computed_ta = False

    if not computed_ta:
        df["rsi"] = _rsi_manual(close, 14).tolist()
        macd_line, sig = _macd_manual(close, 12, 26, 9)
        df["macd"] = (macd_line / (close + 1e-8)).tolist()
        df["macd_signal"] = (sig / (close + 1e-8)).tolist()
        df["atr"] = (_atr_manual(high, low, close, 14) / (close + 1e-8)).tolist()
        df["bb_width"] = (_bbwidth_manual(close, 20, 2) / (close + 1e-8)).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    # Session encoding: cyclical hour-of-day (works for both datetime index and 'time' column)
    if "time" in df.columns:
        time_vals = df["time"].tolist()
        hours = [pd.Timestamp(t).hour for t in time_vals]
    elif isinstance(df.index, pd.DatetimeIndex):
        hours = list(df.index.hour)
    else:
        hours = [0.0] * len(df)
    hour_norm = np.asarray(hours, dtype=np.float64) / 23.0 * 2.0 * np.pi
    df["session_sin"] = np.sin(hour_norm).tolist()
    df["session_cos"] = np.cos(hour_norm).tolist()

    # Volatility regime features using the already-computed raw ATR
    raw_atr = _atr_manual(
        np.asarray(df["high"], dtype=np.float64),
        np.asarray(df["low"], dtype=np.float64),
        np.asarray(df["close"], dtype=np.float64),
        14,
    )
    # ATR percentile: rolling 100-bar rank normalized to [0, 1]
    atr_series = pd.Series(raw_atr.tolist())
    atr_pct = atr_series.rolling(100, min_periods=1).apply(
        lambda x: float(np.sum(x <= x[-1])) / float(len(x)), raw=True
    )
    df["atr_percentile"] = np.asarray(atr_pct, dtype=np.float32).tolist()

    # Bar range ratio: (high - low) / (ATR + 1e-8) — intrabar volatility relative to trend
    high_arr = np.asarray(df["high"], dtype=np.float64)
    low_arr = np.asarray(df["low"], dtype=np.float64)
    df["bar_range_ratio"] = ((high_arr - low_arr) / (raw_atr + 1e-8)).astype(np.float32).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


def _rsi_manual(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0).astype(np.float64)
    loss = np.where(delta < 0, -delta, 0.0).astype(np.float64)
    avg_g = np.asarray(
        pd.Series(gain.tolist()).ewm(alpha=1 / period, min_periods=period).mean(), dtype=np.float64
    )
    avg_l = np.asarray(
        pd.Series(loss.tolist()).ewm(alpha=1 / period, min_periods=period).mean(), dtype=np.float64
    )
    rs = avg_g / (avg_l + 1e-8)
    return np.asarray(1.0 - 1.0 / (1.0 + rs), dtype=np.float64)


def _ema_manual(arr: np.ndarray, span: int) -> np.ndarray:
    return np.asarray(pd.Series(arr.tolist()).ewm(span=span, adjust=False).mean(), dtype=np.float64)


def _macd_manual(close: np.ndarray, fast: int, slow: int, signal: int):
    macd_line = _ema_manual(close, fast) - _ema_manual(close, slow)
    sig_line = _ema_manual(macd_line, signal)
    return macd_line, sig_line


def _atr_manual(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return np.asarray(
        pd.Series(tr.tolist()).ewm(span=period, adjust=False).mean(), dtype=np.float64
    )


def _bbwidth_manual(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    s = pd.Series(close.tolist())
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return np.asarray((upper - lower), dtype=np.float64)


def generate_labels(
    df: pd.DataFrame,
    pair: str,
    horizon: int = LABEL_HORIZON_BARS,
    threshold_pips: float = DIRECTION_THRESHOLD_PIPS,
) -> pd.DataFrame:
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

    close_arr = np.asarray(df["close"], dtype=np.float64)
    high_arr = np.asarray(df["high"], dtype=np.float64)
    low_arr = np.asarray(df["low"], dtype=np.float64)
    n = len(df)

    directions = np.ones(n, dtype=np.int64)  # Default: Sideways
    tp_pips = np.zeros(n, dtype=np.float32)
    sl_pips = np.zeros(n, dtype=np.float32)

    for i in range(n - horizon):
        entry = close_arr[i]
        fut_high = high_arr[i + 1 : i + 1 + horizon].max()
        fut_low = low_arr[i + 1 : i + 1 + horizon].min()
        up_move = (fut_high - entry) / pip_size
        down_move = (entry - fut_low) / pip_size

        # Spread-adjusted magnitudes: TP shrinks (spread cost), SL grows (spread widens stop)
        tp_pips[i] = max(0.0, float(up_move) - spread_pips)
        sl_pips[i] = max(0.0, float(down_move) + spread_pips)

        if up_move >= threshold_pips and up_move > down_move:
            directions[i] = 2  # Up
        elif down_move >= threshold_pips and down_move > up_move:
            directions[i] = 0  # Down
        # else: 1 (Sideways, already default)

    df = df.copy()
    df["direction"] = directions.tolist()
    df["tp_pips"] = tp_pips.tolist()
    df["sl_pips"] = sl_pips.tolist()
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV to returns and volume z-score.

    - open/high/low/close → log-return relative to previous close (removes price-level bias)
    - volume              → z-score over rolling 50-bar window
    - Indicators          → already normalized during compute_indicators
    """
    df = df.copy()
    close = np.asarray(df["close"], dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]

    for col in ["open", "high", "low", "close"]:
        df[col] = np.log(np.asarray(df[col], dtype=np.float64) / (prev + 1e-8)).tolist()

    vol = np.asarray(df["volume"], dtype=np.float64)
    vol_s = pd.Series(vol.tolist())
    df["volume"] = np.asarray(
        (vol_s - vol_s.rolling(50).mean()) / (vol_s.rolling(50).std() + 1e-8), dtype=np.float64
    ).tolist()

    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df
