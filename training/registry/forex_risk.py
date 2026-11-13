"""The Forex ``evaluation.risk_per_trade`` registry setting (single reader, R4-C3).

The validation trade simulation (:mod:`training.training.forex_equity`) compounds equity at
this fraction per trade, so it defines the MaxDD metric: it is fixed for a run (recorded in
the checkpoint run settings and in the Forex quality scheme id), shown in the model card and
used as the ``RiskPct`` default of the generated MQL5 Expert Advisor. The default of 1 %
follows the common 1-2 % money-management rule (1 % for frequent, correlated signals);
values above 5 % are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_RISK_PER_TRADE = 0.01
"""Equity fraction risked at the stop-loss of each simulated trade (registry default)."""
MAX_RISK_PER_TRADE = 0.05
"""Largest accepted ``evaluation.risk_per_trade``; above it the curve's zero floor dominates."""
RISK_PER_TRADE_KEY = "risk_per_trade"
EVALUATION_KEY = "evaluation"


def validate_risk_per_trade(value: Any) -> float:
    """``value`` as a risk fraction in ``(0, MAX_RISK_PER_TRADE]``.

    ``TypeError`` for a non-number, ``ValueError`` for a number out of range.
    """
    if isinstance(value, bool):
        raise TypeError(f"evaluation.risk_per_trade must be a number, got {value!r}")
    try:
        risk = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"evaluation.risk_per_trade must be a number, got {value!r}") from exc
    if not 0.0 < risk <= MAX_RISK_PER_TRADE:
        raise ValueError(
            f"evaluation.risk_per_trade must satisfy 0 < x <= {MAX_RISK_PER_TRADE}, got {risk}"
        )
    return risk


def risk_per_trade(model_info: Mapping[str, Any] | None) -> float:
    """The model's validated ``evaluation.risk_per_trade`` (default 1 %).

    The one reader of the registry setting: validation, the equity model, the quality
    scheme id, the run settings record, the model card and the MQL5 EA template use it.
    """
    evaluation = (model_info or {}).get(EVALUATION_KEY) or {}
    if not isinstance(evaluation, Mapping):
        raise TypeError(f"evaluation must be a mapping, got {type(evaluation).__name__}")
    return validate_risk_per_trade(evaluation.get(RISK_PER_TRADE_KEY, DEFAULT_RISK_PER_TRADE))


def risk_label(risk: float) -> str:
    """``risk`` as a percent label, e.g. ``1%`` or ``0.5%``."""
    return f"{risk * 100.0:g}%"
