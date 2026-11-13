"""MQL5 feature builder of the Forex Expert Advisor, rendered from the feature specification.

The constants (column order, indicator windows, epsilon, warm-up bars) come from
:mod:`data.mt5_feature_spec`, the specification :mod:`data.mt5_features` computes the
training shards with, so the EA cannot drift from the model's input columns. The routines
reproduce :func:`data.mt5_features.compute_indicators` (``pandas_ta_classic`` path) and
:func:`data.mt5_features.normalize_ohlcv` bar for bar over ``lookback + WARMUP_BARS`` closed
bars; ``tests/unit/test_mql5_features.py`` checks a Python mirror of these loops against the
pandas pipeline.

``BuildFeatures`` fills ``x`` (``lookback x FEATURES``, oldest bar first) from the last
``lookback`` closed bars of ``period`` and returns false when the history is too short.
"""

from __future__ import annotations

import re

from data.mt5_feature_spec import (
    EPSILON,
    FEATURE_NAMES,
    FEATURE_WINDOWS,
    FLAT_BAND_TOLERANCE,
    WARMUP_BARS,
)


def column_define(name: str) -> str:
    """MQL5 macro of the model input column of feature ``name``."""
    return f"COL_{name.upper()}"


def _specification_values() -> dict[str, object]:
    """MQL5 macro name and value of each specification constant."""
    w = FEATURE_WINDOWS
    return {
        "WARMUP_BARS": WARMUP_BARS,
        "RSI_PERIOD": w.rsi,
        "MACD_FAST": w.macd.fast,
        "MACD_SLOW": w.macd.slow,
        "MACD_SIGNAL": w.macd.signal,
        "ATR_PERIOD": w.atr,
        "BB_PERIOD": w.bollinger.period,
        "BB_STD": w.bollinger.std,
        "ATR_RANK_WINDOW": w.atr_rank,
        "VOLUME_ZSCORE_WINDOW": w.volume_zscore,
        "FEATURE_EPSILON": EPSILON,
        "FLAT_BAND_TOLERANCE": FLAT_BAND_TOLERANCE,
    }


def _constants() -> list[str]:
    columns = {column_define(name): index for index, name in enumerate(FEATURE_NAMES)}
    return [
        "// Feature specification (data/mt5_feature_spec.py).",
        *(f"#define {name} {value!r}" for name, value in _specification_values().items()),
        "",
        "// Model input column of each feature (data/mt5_feature_spec.py FEATURE_NAMES order).",
        *(f"#define {name} {index}" for name, index in columns.items()),
        "",
    ]


_ROUTINES = """\
// SMA-seeded recursive average (pandas_ta_classic rma and ema): out[first + length - 1] is
// the mean of values[first .. first + length - 1], later bars follow
// out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]. Returns the seed bar; bars before
// it stay 0 (the pipeline's fillna).
int SeededAverage(const double &values[], const int first, const int length,
                  const double alpha, const int count, double &out[])
  {
   ArrayResize(out, count);
   ArrayInitialize(out, 0.0);
   int seed = first + length - 1;
   if(first < 0 || seed >= count)
      return(count);
   double sum = 0.0;
   for(int k = first; k <= seed; k++)
      sum += values[k];
   out[seed] = sum / length;
   for(int i = seed + 1; i < count; i++)
      out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1];
   return(seed);
  }

// Mean and standard deviation (divisor window - ddof) of values[last - window + 1 .. last].
double WindowStd(const double &values[], const int last, const int window, const int ddof,
                 double &mean)
  {
   double sum = 0.0;
   for(int k = last - window + 1; k <= last; k++)
      sum += values[k];
   mean = sum / window;
   double squares = 0.0;
   for(int k = last - window + 1; k <= last; k++)
      squares += (values[k] - mean) * (values[k] - mean);
   return(MathSqrt(squares / (window - ddof)));
  }

void PutFeature(matrixf &x, const int row, const int column, const double value)
  {
   double finite = MathIsValidNumber(value) ? value : 0.0;
   x[row][column] = (float)finite;
  }

// Reproduces data/mt5_features.py compute_indicators + normalize_ohlcv for the last
// `lookback` closed bars of `period` (oldest first), computed over lookback + WARMUP_BARS
// closed bars.
bool BuildFeatures(const ENUM_TIMEFRAMES period, const int lookback, matrixf &x)
  {
   int count = lookback + WARMUP_BARS;
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   if(CopyRates(_Symbol, period, 1, count, rates) != count)
     {
      PrintFormat("CopyRates(%s, %s) returned fewer than %d bars, error %d", _Symbol,
                  EnumToString(period), count, GetLastError());
      return(false);
     }

   double bar_open[], bar_high[], bar_low[], bar_close[], bar_volume[];
   double gain[], loss[], range_true[], range_raw[];
   ArrayResize(bar_open, count);
   ArrayResize(bar_high, count);
   ArrayResize(bar_low, count);
   ArrayResize(bar_close, count);
   ArrayResize(bar_volume, count);
   ArrayResize(gain, count);
   ArrayResize(loss, count);
   ArrayResize(range_true, count);
   ArrayResize(range_raw, count);
   for(int i = 0; i < count; i++)
     {
      bar_open[i] = rates[i].open;
      bar_high[i] = rates[i].high;
      bar_low[i] = rates[i].low;
      bar_close[i] = rates[i].close;
      bar_volume[i] = (double)rates[i].tick_volume;
      double prev_close = (i > 0) ? rates[i - 1].close : rates[0].close;
      range_raw[i] = MathMax(bar_high[i] - bar_low[i],
                             MathMax(MathAbs(bar_high[i] - prev_close),
                                     MathAbs(bar_low[i] - prev_close)));
      gain[i] = 0.0;
      loss[i] = 0.0;
      range_true[i] = 0.0;
      if(i > 0)
        {
         double change = bar_close[i] - bar_close[i - 1];
         gain[i] = (change > 0.0) ? change : 0.0;
         loss[i] = (change < 0.0) ? -change : 0.0;
         range_true[i] = MathMax(MathAbs(bar_high[i] - bar_low[i]),
                                 MathMax(MathAbs(bar_high[i] - prev_close),
                                         MathAbs(prev_close - bar_low[i])));
        }
     }

   // RSI and ATR: Wilder averages from bar 1; MACD and signal: EMAs.
   double gain_avg[], loss_avg[], atr[], fast[], slow[], signal_line[];
   int rsi_start = SeededAverage(gain, 1, RSI_PERIOD, 1.0 / RSI_PERIOD, count, gain_avg);
   SeededAverage(loss, 1, RSI_PERIOD, 1.0 / RSI_PERIOD, count, loss_avg);
   int atr_start = SeededAverage(range_true, 1, ATR_PERIOD, 1.0 / ATR_PERIOD, count, atr);
   SeededAverage(bar_close, 0, MACD_FAST, 2.0 / (MACD_FAST + 1), count, fast);
   int macd_start = SeededAverage(bar_close, 0, MACD_SLOW, 2.0 / (MACD_SLOW + 1), count, slow);
   double macd[];
   ArrayResize(macd, count);
   for(int i = 0; i < count; i++)
      macd[i] = (i >= macd_start) ? fast[i] - slow[i] : 0.0;
   int signal_start = SeededAverage(macd, macd_start, MACD_SIGNAL, 2.0 / (MACD_SIGNAL + 1),
                                    count, signal_line);

   // Regime ATR: EMA of the true range, bar 0 seeded with its own range.
   double atr_raw[];
   ArrayResize(atr_raw, count);
   double atr_alpha = 2.0 / (ATR_PERIOD + 1);
   atr_raw[0] = range_raw[0];
   for(int i = 1; i < count; i++)
      atr_raw[i] = atr_alpha * range_raw[i] + (1.0 - atr_alpha) * atr_raw[i - 1];

   if(!x.Init((ulong)lookback, FEATURES))
      return(false);
   int offset = count - lookback;
   for(int row = 0; row < lookback; row++)
     {
      int i = offset + row;
      double prev_close = (i > 0) ? bar_close[i - 1] : bar_close[0];
      double scale = bar_close[i] + FEATURE_EPSILON;
      PutFeature(x, row, COL_OPEN, MathLog(bar_open[i] / (prev_close + FEATURE_EPSILON)));
      PutFeature(x, row, COL_HIGH, MathLog(bar_high[i] / (prev_close + FEATURE_EPSILON)));
      PutFeature(x, row, COL_LOW, MathLog(bar_low[i] / (prev_close + FEATURE_EPSILON)));
      PutFeature(x, row, COL_CLOSE, MathLog(bar_close[i] / (prev_close + FEATURE_EPSILON)));

      double mean = 0.0;
      double zscore = 0.0;
      if(i >= VOLUME_ZSCORE_WINDOW - 1)
        {
         double deviation = WindowStd(bar_volume, i, VOLUME_ZSCORE_WINDOW, 1, mean);
         zscore = (bar_volume[i] - mean) / (deviation + FEATURE_EPSILON);
        }
      PutFeature(x, row, COL_VOLUME, zscore);

      double rsi = 0.0;
      double denominator = gain_avg[i] + loss_avg[i];
      if(i >= rsi_start && denominator != 0.0)
         rsi = gain_avg[i] / denominator;
      PutFeature(x, row, COL_RSI, rsi);
      PutFeature(x, row, COL_MACD, (i >= macd_start) ? macd[i] / scale : 0.0);
      PutFeature(x, row, COL_MACD_SIGNAL, (i >= signal_start) ? signal_line[i] / scale : 0.0);
      PutFeature(x, row, COL_ATR, (i >= atr_start) ? atr[i] / scale : 0.0);

      double band = 0.0;
      if(i >= BB_PERIOD - 1)
        {
         double deviation = WindowStd(bar_close, i, BB_PERIOD, 0, mean);
         band = (mean + BB_STD * deviation) - (mean - BB_STD * deviation);
         if(MathAbs(band) <= FLAT_BAND_TOLERANCE)
            band = 0.0;
        }
      PutFeature(x, row, COL_BB_WIDTH, band / scale);

      MqlDateTime when;
      TimeToStruct(rates[i].time, when);
      double angle = when.hour / 23.0 * 2.0 * M_PI;
      PutFeature(x, row, COL_SESSION_SIN, MathSin(angle));
      PutFeature(x, row, COL_SESSION_COS, MathCos(angle));

      int first = MathMax(0, i - ATR_RANK_WINDOW + 1);
      int below = 0;
      for(int k = first; k <= i; k++)
        {
         if(atr_raw[k] <= atr_raw[i])
            below++;
        }
      PutFeature(x, row, COL_ATR_PERCENTILE, (double)below / (double)(i - first + 1));
      PutFeature(x, row, COL_BAR_RANGE_RATIO,
                 (bar_high[i] - bar_low[i]) / (atr_raw[i] + FEATURE_EPSILON));
     }
   return(true);
  }
"""


class FeatureSpecMismatch(ValueError):
    """The MQL5 feature builder does not write every column of the specification."""


def _check_columns(source: str) -> None:
    """Every specification column is written exactly once by ``BuildFeatures``."""
    written = re.findall(r"PutFeature\(x, row, (COL_[A-Z_]+),", source)
    expected = [column_define(name) for name in FEATURE_NAMES]
    if sorted(written) != sorted(expected):
        raise FeatureSpecMismatch(
            f"BuildFeatures writes {sorted(written)}, the specification has {sorted(expected)}"
        )


def render_feature_builder() -> list[str]:
    """The MQL5 lines of the feature constants and ``BuildFeatures``."""
    _check_columns(_ROUTINES)
    return [*_constants(), *_ROUTINES.rstrip("\n").split("\n"), ""]
