"""The MQL5 EA feature builder against ``data.mt5_features`` (R4-FU item 1).

``_mql5_build_features`` mirrors the generated ``BuildFeatures`` loop for loop (the MQL5 code
cannot be compiled here), so its agreement with the pandas pipeline that wrote the training
shards checks the port; the golden file pins the generated EA text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.mt5_feature_spec import (
    EPSILON,
    FEATURE_NAMES,
    FEATURE_WINDOWS,
    FLAT_BAND_TOLERANCE,
    WARMUP_BARS,
)
from data.mt5_features import FEATURES, compute_indicators, normalize_ohlcv
from export.mt5_signal import generate_mql5_ea
from models.forex_predictor import FOREX_FEATURES_PER_BAR
from training.export import mql5_features
from training.export.mql5_features import (
    FeatureSpecMismatch,
    column_define,
    render_feature_builder,
)

GOLDEN_EA = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "LemGendaryForexEA.mq5"
"""Regenerate with ``python -m export.mt5_signal --mode mql5_ea --out <this path>``."""
_LOOKBACK = 168


def _bars(count: int, seed: int = 7) -> pd.DataFrame:
    """Deterministic hourly OHLCV bars with MT5 tick volumes."""
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0.0, 0.001, count)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1.0 + rng.normal(0.0, 0.0002, count))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.0005, count)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.0005, count)))
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 5000, count),
        }
    )


def _seeded_average(
    values: Sequence[float], first: int, length: int, alpha: float, out: list[float]
) -> int:
    """``SeededAverage``: SMA seed, then the recursive average; returns the seed bar."""
    count = len(values)
    out[:] = [0.0] * count
    seed = first + length - 1
    if first < 0 or seed >= count:
        return count
    out[seed] = sum(values[first : seed + 1]) / length
    for i in range(seed + 1, count):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return seed


def _window_std(values: Sequence[float], last: int, window: int, ddof: int) -> tuple[float, float]:
    """``WindowStd``: (standard deviation, mean) of the window ending at ``last``."""
    part = values[last - window + 1 : last + 1]
    mean = sum(part) / window
    squares = sum((v - mean) * (v - mean) for v in part)
    return math.sqrt(squares / (window - ddof)), mean


def _bar_series(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Bar columns, price changes and true ranges as ``BuildFeatures`` first collects them."""
    bars = {name: [float(v) for v in frame[name]] for name in ("open", "high", "low", "close")}
    bars["volume"] = [float(v) for v in frame["volume"]]
    bars["hour"] = [float(pd.Timestamp(t).hour) for t in frame["time"]]
    close, high, low = bars["close"], bars["high"], bars["low"]
    prev = [close[0], *close[:-1]]
    changes = [0.0] + [close[i] - close[i - 1] for i in range(1, len(close))]
    bars["gain"] = [max(0.0, change) for change in changes]
    bars["loss"] = [max(0.0, -change) for change in changes]
    bars["range_raw"] = [
        max(h - lo, abs(h - p), abs(lo - p)) for h, lo, p in zip(high, low, prev, strict=True)
    ]
    bars["range_true"] = [0.0] + [
        max(abs(high[i] - low[i]), abs(high[i] - prev[i]), abs(prev[i] - low[i]))
        for i in range(1, len(close))
    ]
    return bars


def _indicators(bars: dict[str, list[float]]) -> tuple[dict[str, list[float]], dict[str, int]]:
    """The recursive averages of ``BuildFeatures`` and the bar each becomes valid."""
    w = FEATURE_WINDOWS
    series: dict[str, list[float]] = {}
    start: dict[str, int] = {}
    for name in ("gain_avg", "loss_avg", "atr", "fast", "slow", "signal"):
        series[name] = []
    start["rsi"] = _seeded_average(bars["gain"], 1, w.rsi, 1.0 / w.rsi, series["gain_avg"])
    _seeded_average(bars["loss"], 1, w.rsi, 1.0 / w.rsi, series["loss_avg"])
    start["atr"] = _seeded_average(bars["range_true"], 1, w.atr, 1.0 / w.atr, series["atr"])
    macd = w.macd
    _seeded_average(bars["close"], 0, macd.fast, 2.0 / (macd.fast + 1), series["fast"])
    start["macd"] = _seeded_average(
        bars["close"], 0, macd.slow, 2.0 / (macd.slow + 1), series["slow"]
    )
    series["macd"] = [
        fast - slow if i >= start["macd"] else 0.0
        for i, (fast, slow) in enumerate(zip(series["fast"], series["slow"], strict=True))
    ]
    start["signal"] = _seeded_average(
        series["macd"], start["macd"], macd.signal, 2.0 / (macd.signal + 1), series["signal"]
    )
    alpha = 2.0 / (w.atr + 1)
    series["atr_raw"] = [bars["range_raw"][0]]
    for value in bars["range_raw"][1:]:
        series["atr_raw"].append(alpha * value + (1.0 - alpha) * series["atr_raw"][-1])
    return series, start


def _price_columns(bars: dict[str, list[float]], i: int) -> dict[str, float]:
    """Log returns against the previous close and the volume z-score of bar ``i``."""
    close = bars["close"]
    prev_close = close[i - 1] if i > 0 else close[0]
    row = {
        name: math.log(bars[name][i] / (prev_close + EPSILON))
        for name in ("open", "high", "low", "close")
    }
    row["volume"] = 0.0
    window = FEATURE_WINDOWS.volume_zscore
    if i >= window - 1:
        deviation, mean = _window_std(bars["volume"], i, window, 1)
        row["volume"] = (bars["volume"][i] - mean) / (deviation + EPSILON)
    return row


def _indicator_columns(
    bars: dict[str, list[float]], ind: dict[str, list[float]], start: dict[str, int], i: int
) -> dict[str, float]:
    """RSI, MACD, MACD signal, ATR and Bollinger width of bar ``i``."""
    scale = bars["close"][i] + EPSILON
    denominator = ind["gain_avg"][i] + ind["loss_avg"][i]
    rsi_ready = i >= start["rsi"] and denominator != 0.0
    row = {"rsi": ind["gain_avg"][i] / denominator if rsi_ready else 0.0}
    for name, key in (("macd", "macd"), ("macd_signal", "signal"), ("atr", "atr")):
        row[name] = ind[key][i] / scale if i >= start[key] else 0.0
    band = 0.0
    bollinger = FEATURE_WINDOWS.bollinger
    if i >= bollinger.period - 1:
        deviation, mean = _window_std(bars["close"], i, bollinger.period, 0)
        band = (mean + bollinger.std * deviation) - (mean - bollinger.std * deviation)
        band = 0.0 if abs(band) <= FLAT_BAND_TOLERANCE else band
    row["bb_width"] = band / scale
    return row


def _regime_columns(bars: dict[str, list[float]], atr_raw: list[float], i: int) -> dict[str, float]:
    """Session encoding, ATR rank and bar range ratio of bar ``i``."""
    angle = bars["hour"][i] / 23.0 * 2.0 * math.pi
    first = max(0, i - FEATURE_WINDOWS.atr_rank + 1)
    below = sum(1 for k in range(first, i + 1) if atr_raw[k] <= atr_raw[i])
    return {
        "session_sin": math.sin(angle),
        "session_cos": math.cos(angle),
        "atr_percentile": below / (i - first + 1),
        "bar_range_ratio": (bars["high"][i] - bars["low"][i]) / (atr_raw[i] + EPSILON),
    }


def _mql5_build_features(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    """``BuildFeatures`` in Python: the last ``lookback`` feature rows of ``frame``."""
    bars = _bar_series(frame)
    ind, start = _indicators(bars)
    count = len(bars["close"])
    rows = [
        {
            **_price_columns(bars, i),
            **_indicator_columns(bars, ind, start, i),
            **_regime_columns(bars, ind["atr_raw"], i),
        }
        for i in range(count - lookback, count)
    ]
    return np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)


def test_spec_is_the_pipeline_column_order() -> None:
    """The EA is generated from the columns ``data.mt5_features`` writes, at model width."""
    assert FEATURES == list(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == FOREX_FEATURES_PER_BAR


def test_columns_are_rendered_in_spec_order() -> None:
    """``COL_<NAME>`` is the feature's index in the specification."""
    source = "\n".join(render_feature_builder())
    for index, name in enumerate(FEATURE_NAMES):
        assert f"#define {column_define(name)} {index}\n" in source
    assert f"#define WARMUP_BARS {WARMUP_BARS}\n" in source
    assert f"#define RSI_PERIOD {FEATURE_WINDOWS.rsi}\n" in source


def test_mql5_port_matches_the_pandas_pipeline() -> None:
    """The mirrored EA loops give the training features of the last ``lookback`` bars."""
    pytest.importorskip("pandas_ta_classic")
    frame = _bars(_LOOKBACK + WARMUP_BARS)
    expected = normalize_ohlcv(compute_indicators(frame))[FEATURES].to_numpy(dtype=np.float32)
    got = _mql5_build_features(frame, _LOOKBACK)
    np.testing.assert_allclose(got, expected[-_LOOKBACK:], rtol=1e-5, atol=1e-6)


def test_a_missing_column_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A builder that does not write every specified column is not rendered."""
    routines = mql5_features.__dict__["_ROUTINES"]
    monkeypatch.setattr(
        mql5_features, "_ROUTINES", routines.replace("COL_BAR_RANGE_RATIO", "COL_ATR")
    )
    with pytest.raises(FeatureSpecMismatch):
        render_feature_builder()


def test_generated_ea_matches_the_golden_file(tmp_path: Path) -> None:
    """The EA written by the generator is the reviewed golden text."""
    written = generate_mql5_ea(str(tmp_path / "LemGendaryForexEA.mq5"))
    assert written.read_text(encoding="utf-8") == GOLDEN_EA.read_text(encoding="utf-8")
    assert "BuildFeatures is not ported" not in written.read_text(encoding="utf-8")
