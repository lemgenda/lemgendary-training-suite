"""MQL5 Expert Advisor template for the ForexPredictor ONNX artefact (R4-C1).

The template is rendered from the model's registry entry, so its constants come from the one
source of each value:

* the ONNX inputs (one ``tf_<minutes>`` matrix of ``lookback x features`` per active timeframe,
  then ``pair_idx``) from ``kwargs.active_timeframes``, ``kwargs.in_features`` and
  :data:`data.forex_spec.TIMEFRAME_LOOKBACK`; the outputs ``direction_logits [1, 3]`` and
  ``magnitude [1, 2]`` (take-profit and stop-loss in model units), all FP32 except the int64
  pair index;
* the pair indices from :data:`data.forex_spec.PAIR_INDEX` and the price of one magnitude
  unit: the pip size of the downloaded ticker (:data:`data.forex_spec.MT5_SYMBOLS`, the name
  the labels were generated for) times ``PAIR_PIP_SCALE`` (the label scaling of
  ``data.forex_dataset``);
* the ``RiskPct`` default from ``evaluation.risk_per_trade``
  (:func:`training.registry.forex_risk.risk_per_trade`).

The EA loads the single-file IR 9 model with ``OnnxCreate`` (``MQL5\\Files``), fixes the batch
axis to 1 with ``OnnxSetInputShape`` / ``OnnxSetOutputShape``, runs it with ``OnnxRun`` on every
new bar and trades through ``CTrade`` with risk-based position sizing. The feature builder
(``BuildFeatures``) is rendered by :mod:`training.export.mql5_features` from the feature
specification :mod:`data.mt5_feature_spec` that :mod:`data.mt5_features` computes the
training shards with; the model's ``in_features`` must equal that specification's width.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from data.forex_spec import MT5_SYMBOLS, PAIR_INDEX, PAIR_PIP_SCALE, TIMEFRAME_LOOKBACK
from data.mt5_feature_spec import FEATURE_NAMES
from data.mt5_features import pair_pip_size
from models.forex_predictor import FOREX_FEATURES_PER_BAR
from training.export.mql5_features import render_feature_builder
from training.export.mt5_signal import DEFAULT_TIMEFRAMES, DIRECTION_LABELS
from training.registry.forex_risk import risk_per_trade

MQL5_PERIODS: dict[int, str] = {
    1: "PERIOD_M1",
    5: "PERIOD_M5",
    15: "PERIOD_M15",
    30: "PERIOD_M30",
    60: "PERIOD_H1",
    240: "PERIOD_H4",
    1440: "PERIOD_D1",
    10080: "PERIOD_W1",
}
"""MQL5 ``ENUM_TIMEFRAMES`` name of each timeframe in minutes."""
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_MAGIC_NUMBER = 20260724


@dataclass(frozen=True)
class EaSymbol:
    """One tradable symbol: its name, the model's pair index and the price of a model unit."""

    name: str
    pair_index: int
    unit_price: float


@dataclass(frozen=True)
class EaSpec:
    """Everything the EA template needs (see the module docstring for the sources)."""

    model_file: str
    timeframes: tuple[int, ...]
    features: int
    risk_pct: float
    symbols: tuple[EaSymbol, ...]
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    magic_number: int = DEFAULT_MAGIC_NUMBER

    def lookback(self, minutes: int) -> int:
        """Bars per sample of the ``minutes`` timeframe."""
        return TIMEFRAME_LOOKBACK.get(minutes, 48)


def _symbols(pairs: Sequence[str]) -> tuple[EaSymbol, ...]:
    """The model's pairs and their broker aliases with index and model-unit price."""
    names = [name for name in PAIR_INDEX if name in pairs or _canonical(name) in pairs]
    return tuple(EaSymbol(name, PAIR_INDEX[name], _unit_price(name)) for name in names)


def _unit_price(name: str) -> float:
    """Price of one model magnitude unit: label pip size times the pair's pip scale."""
    ticker = MT5_SYMBOLS[PAIR_INDEX[name]]
    return pair_pip_size(ticker) * PAIR_PIP_SCALE.get(name, 1.0)


def _canonical(name: str) -> str:
    """The canonical pair of a broker alias (the alias shares its index)."""
    index = PAIR_INDEX[name]
    return next(pair for pair, idx in PAIR_INDEX.items() if idx == index)


def ea_spec(model_info: Mapping[str, Any], model_file: str) -> EaSpec:
    """The EA spec of a ForexPredictor registry entry whose ONNX file is ``model_file``."""
    kwargs = model_info.get("kwargs") or {}
    timeframes = tuple(int(tf) for tf in kwargs.get("active_timeframes") or DEFAULT_TIMEFRAMES)
    unknown = [tf for tf in timeframes if tf not in MQL5_PERIODS]
    if unknown:
        raise ValueError(f"No MQL5 timeframe for {unknown} minutes")
    features = int(kwargs.get("in_features") or FOREX_FEATURES_PER_BAR)
    if features != len(FEATURE_NAMES):
        raise ValueError(
            f"in_features {features} differs from the {len(FEATURE_NAMES)} specified features"
        )
    pairs = list(kwargs.get("pairs") or PAIR_INDEX)
    return EaSpec(
        model_file=model_file,
        timeframes=timeframes,
        features=features,
        risk_pct=risk_per_trade(model_info) * 100.0,
        symbols=_symbols(pairs),
    )


def _class_index(label: str) -> int:
    return next(idx for idx, name in DIRECTION_LABELS.items() if name == label)


def _array(values: Sequence[str]) -> str:
    return "{" + ", ".join(values) + "}"


def _header(spec: EaSpec) -> list[str]:
    tf_names = ", ".join(f"tf_{tf}" for tf in spec.timeframes)
    return [
        "//+------------------------------------------------------------------+",
        "//| LemGendary Forex EA (ForexPredictor ONNX)                        |",
        "//| Generated by training/export/mql5_ea.py from the model registry; |",
        "//| regenerate instead of editing by hand.                           |",
        "//+------------------------------------------------------------------+",
        '#property copyright "LemGendary AI"',
        '#property version   "2.00"',
        "",
        "#include <Trade\\Trade.mqh>",
        "",
        f"// Model inputs: {tf_names} [1, lookback, {spec.features}] float, pair_idx [1] long.",
        "// Model outputs: direction_logits [1, 3] float, magnitude [1, 2] float.",
        f'input string ModelFile     = "{spec.model_file}"; // single file in MQL5\\Files',
        f"input double RiskPct       = {spec.risk_pct:.2f};   // % of balance risked at the SL",
        f"input double MinConfidence = {spec.min_confidence:.2f};  // minimum class probability",
        f"input ulong  MagicNumber   = {spec.magic_number};",
        "",
        f"#define TF_COUNT {len(spec.timeframes)}",
        f"#define FEATURES {spec.features}",
        f"#define CLASS_SELL {_class_index('SELL')}",
        f"#define CLASS_HOLD {_class_index('HOLD')}",
        f"#define CLASS_BUY {_class_index('BUY')}",
        "",
        "ENUM_TIMEFRAMES TF_PERIODS[TF_COUNT] = "
        + _array([MQL5_PERIODS[tf] for tf in spec.timeframes])
        + ";",
        "int TF_LOOKBACK[TF_COUNT] = "
        + _array([str(spec.lookback(tf)) for tf in spec.timeframes])
        + ";",
        "string PAIR_SYMBOLS[] = " + _array([f'"{s.name}"' for s in spec.symbols]) + ";",
        "long PAIR_INDICES[] = " + _array([str(s.pair_index) for s in spec.symbols]) + ";",
        "double PAIR_UNIT_PRICE[] = " + _array([f"{s.unit_price:g}" for s in spec.symbols]) + ";",
        "",
        "long     g_model = INVALID_HANDLE;",
        "CTrade   g_trade;",
        "datetime g_last_bar = 0;",
        "long     g_pair_index = -1;",
        "double   g_unit_price = 0.0;",
        "",
    ]


def _set_shapes(spec: EaSpec) -> list[str]:
    lines = ["bool SetShapes()", "  {"]
    for i, tf in enumerate(spec.timeframes):
        lines += [
            f"   ulong shape_{i}[] = {{1, {spec.lookback(tf)}, {spec.features}}};",
            f"   if(!OnnxSetInputShape(g_model, {i}, shape_{i}))",
            "      return(false);",
        ]
    lines += [
        "   ulong pair_shape[] = {1};",
        "   if(!OnnxSetInputShape(g_model, TF_COUNT, pair_shape))",
        "      return(false);",
        "   ulong direction_shape[] = {1, 3};",
        "   ulong magnitude_shape[] = {1, 2};",
        "   return(OnnxSetOutputShape(g_model, 0, direction_shape) &&",
        "          OnnxSetOutputShape(g_model, 1, magnitude_shape));",
        "  }",
        "",
    ]
    return lines


_LIFECYCLE = """\
bool ResolveSymbol()
  {
   for(int i = 0; i < ArraySize(PAIR_SYMBOLS); i++)
     {
      if(StringFind(_Symbol, PAIR_SYMBOLS[i]) == 0)
        {
         g_pair_index = PAIR_INDICES[i];
         g_unit_price = PAIR_UNIT_PRICE[i];
         return(true);
        }
     }
   return(false);
  }

int OnInit()
  {
   if(!ResolveSymbol())
     {
      PrintFormat("%s is not one of the model's pairs", _Symbol);
      return(INIT_FAILED);
     }
   g_model = OnnxCreate(ModelFile, ONNX_DEFAULT);
   if(g_model == INVALID_HANDLE)
     {
      PrintFormat("OnnxCreate(%s) failed, error %d", ModelFile, GetLastError());
      return(INIT_FAILED);
     }
   if(!SetShapes())
     {
      PrintFormat("Setting the ONNX shapes failed, error %d", GetLastError());
      OnnxRelease(g_model);
      g_model = INVALID_HANDLE;
      return(INIT_FAILED);
     }
   g_trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(g_model != INVALID_HANDLE)
      OnnxRelease(g_model);
   g_model = INVALID_HANDLE;
  }

double LotsForRisk(const double stop_distance)
  {
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(tick_value <= 0.0 || tick_size <= 0.0 || step <= 0.0 || stop_distance <= 0.0)
      return(0.0);
   double risk_money = AccountInfoDouble(ACCOUNT_BALANCE) * RiskPct / 100.0;
   double loss_per_lot = stop_distance / tick_size * tick_value;
   double lots = MathFloor(risk_money / loss_per_lot / step) * step;
   if(lots < SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
      return(0.0);
   return(MathMin(lots, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX)));
  }

void Trade(const int signal, const double take_units, const double stop_units)
  {
   double take = take_units * g_unit_price;
   double stop = stop_units * g_unit_price;
   double lots = LotsForRisk(stop);
   if(lots <= 0.0 || take <= 0.0)
      return;
   if(signal == CLASS_BUY)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      g_trade.Buy(lots, _Symbol, ask, NormalizeDouble(ask - stop, _Digits),
                  NormalizeDouble(ask + take, _Digits), "LemGendary");
     }
   else
      if(signal == CLASS_SELL)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         g_trade.Sell(lots, _Symbol, bid, NormalizeDouble(bid + stop, _Digits),
                      NormalizeDouble(bid - take, _Digits), "LemGendary");
        }
  }

"""


def _on_tick(spec: EaSpec) -> list[str]:
    count = len(spec.timeframes)
    lines = [
        "void OnTick()",
        "  {",
        "   datetime bar = iTime(_Symbol, TF_PERIODS[0], 0);",
        "   if(bar == 0 || bar == g_last_bar)",
        "      return;",
        "   g_last_bar = bar;",
        "   if(PositionSelect(_Symbol))",
        "      return;",
        "",
    ]
    for i in range(count):
        lines += [
            f"   matrixf x_{i}(TF_LOOKBACK[{i}], FEATURES);",
            f"   if(!BuildFeatures(TF_PERIODS[{i}], TF_LOOKBACK[{i}], x_{i}))",
            "      return;",
        ]
    inputs = ", ".join(f"x_{i}" for i in range(count))
    lines += [
        "   long pair[1];",
        "   pair[0] = g_pair_index;",
        "   vectorf direction(3);",
        "   vectorf magnitude(2);",
        f"   if(!OnnxRun(g_model, ONNX_DEFAULT, {inputs}, pair, direction, magnitude))",
        "     {",
        '      PrintFormat("OnnxRun failed, error %d", GetLastError());',
        "      return;",
        "     }",
        "",
        "   // Softmax over the three direction logits.",
        "   double top = MathMax(direction[0], MathMax(direction[1], direction[2]));",
        "   double weights[3];",
        "   double total = 0.0;",
        "   for(int k = 0; k < 3; k++)",
        "     {",
        "      weights[k] = MathExp(direction[k] - top);",
        "      total += weights[k];",
        "     }",
        "   int signal = ArrayMaximum(weights);",
        "   if(signal == CLASS_HOLD || weights[signal] / total < MinConfidence)",
        "      return;",
        "   Trade(signal, magnitude[0], magnitude[1]);",
        "  }",
        "//+------------------------------------------------------------------+",
    ]
    return lines


def render_mql5_ea(spec: EaSpec) -> str:
    """The ``.mq5`` source of the Expert Advisor for ``spec``."""
    lines = [
        *_header(spec),
        *render_feature_builder(),
        *_set_shapes(spec),
        _LIFECYCLE.rstrip("\n"),
        "",
        *_on_tick(spec),
    ]
    return "\n".join(lines) + "\n"
