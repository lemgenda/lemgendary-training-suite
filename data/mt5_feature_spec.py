"""Per-bar feature specification of the MT5 Forex model input (single source of truth).

The column order, indicator windows and numeric guards used by
:mod:`data.mt5_features` (training shards and the Python live signal) and rendered into the
MQL5 Expert Advisor feature builder (:mod:`training.export.mql5_features`).

The indicator columns follow the ``pandas_ta_classic`` path of
:func:`data.mt5_features.compute_indicators` (the package is pinned in ``requirements.txt``):
RSI and ATR are Wilder averages seeded with an SMA, MACD and its signal are EMAs seeded with
an SMA, the Bollinger width uses the population standard deviation. The regime columns use
the NumPy ATR (EMA of the true range, first bar seeded with its own range).
"""

from __future__ import annotations

from dataclasses import dataclass

_PRICE_INDICATOR_NAMES: tuple[str, ...] = (
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
)
"""OHLCV (5) and indicator (5) columns."""

FEATURE_NAMES: tuple[str, ...] = (
    *_PRICE_INDICATOR_NAMES,
    "session_sin",
    "session_cos",
    "atr_percentile",
    "bar_range_ratio",
)
"""Model input columns in order: OHLCV, indicators, session (2), volatility regime (2)."""


@dataclass(frozen=True)
class MacdWindows:
    """Fast EMA, slow EMA and signal EMA bar counts."""

    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass(frozen=True)
class BollingerWindow:
    """Bollinger band bar count and standard-deviation multiplier."""

    period: int = 20
    std: float = 2.0


@dataclass(frozen=True)
class FeatureWindows:
    """Bar counts and multipliers of the per-bar indicators."""

    rsi: int = 14
    macd: MacdWindows = MacdWindows()
    atr: int = 14
    bollinger: BollingerWindow = BollingerWindow()
    atr_rank: int = 100
    volume_zscore: int = 50


FEATURE_WINDOWS = FeatureWindows()
EPSILON = 1e-8
"""Added to every divisor (close, ATR, standard deviation, previous close)."""
FLAT_BAND_TOLERANCE = 1e-12
"""Bollinger band width at or below this is a flat window and becomes 0 (pandas_ta_classic)."""
WARMUP_BARS = 300
"""Bars fetched before the lookback window so the recursive averages have converged."""
