"""Forex ``evaluation.risk_per_trade`` registry setting (R4-C3).

One reader (:mod:`training.registry.forex_risk`) validates ``0 < x <= 0.05`` (default 1 %);
the validation equity model, the Forex quality scheme id and the model card use it.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from training.core_loop import load_model_info
from training.governance.quality_score import (
    FOREX_SCHEME,
    anchor_is_comparable,
    forex_scheme,
    quality_scheme,
)
from training.model_card import build_model_readme
from training.registry.forex_risk import (
    DEFAULT_RISK_PER_TRADE,
    MAX_RISK_PER_TRADE,
    risk_label,
    risk_per_trade,
    validate_risk_per_trade,
)
from training.training.forex_equity import max_drawdown_pct, risk_multiple
from training.training.val_accumulators import ValAccumulators
from training.training.val_summary import summarize

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)


def test_default_is_one_percent() -> None:
    """Without the setting the equity model risks 1 % per trade."""
    assert DEFAULT_RISK_PER_TRADE == 0.01
    assert risk_per_trade({}) == 0.01
    assert risk_per_trade(None) == 0.01
    assert risk_per_trade({"evaluation": {}}) == 0.01


@pytest.mark.parametrize("value", [0.0, -0.01, 0.0501, 1.0, "high", None, True, math.nan])
def test_invalid_values_are_rejected(value: Any) -> None:
    """Out of range is a ValueError, a non-number a TypeError."""
    with pytest.raises((TypeError, ValueError), match="risk_per_trade"):
        validate_risk_per_trade(value)


def test_upper_bound_is_accepted() -> None:
    """5 % is the largest accepted value; numeric strings are read as numbers."""
    assert validate_risk_per_trade(MAX_RISK_PER_TRADE) == 0.05
    assert risk_per_trade({"evaluation": {"risk_per_trade": "0.02"}}) == 0.02


def test_evaluation_must_be_a_mapping() -> None:
    """The ``evaluation`` block is a mapping."""
    with pytest.raises(TypeError, match="evaluation"):
        risk_per_trade({"evaluation": [0.01]})


def test_registry_entries_are_valid() -> None:
    """The forex entry sets the default explicitly; every entry passes validation."""
    assert _REGISTRY["forex_predictor"]["evaluation"]["risk_per_trade"] == 0.01
    for key, entry in _REGISTRY.items():
        if key != "_registry_metadata" and isinstance(entry, dict):
            risk_per_trade(entry)


def test_load_model_info_rejects_an_invalid_value(tmp_path: Path) -> None:
    """The training entry point refuses a registry entry with an invalid risk."""
    registry = {"forex_predictor": {"evaluation": {"risk_per_trade": 0.2}}}
    (tmp_path / "unified_models_v2.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_model_info(tmp_path, {}, "forex_predictor")


def _summary(risk: float | None) -> dict[str, float]:
    acc = ValAccumulators()
    acc.direction.has_forex_metrics = True
    acc.direction.total_forex_samples = 4
    acc.direction.correct_forex_samples = 1
    acc.direction.total_forex_entropy = 3.6
    trades = [(20.0, 10.0, True), (20.0, 10.0, False), (20.0, 10.0, False)]
    acc.direction.total_trades = len(trades)
    acc.direction.winning_trades = 1
    for tp, sl, won in trades:
        acc.trades.trade_returns.append(tp if won else -sl)
        multiple = risk_multiple(tp, sl, won)
        assert multiple is not None
        acc.trades.trade_risk_multiples.append(multiple)
    info = {} if risk is None else {"evaluation": {"risk_per_trade": risk}}
    ctx = SimpleNamespace(
        model_name="forex_predictor",
        models=SimpleNamespace(model=None),
        run=SimpleNamespace(model_info=info),
    )
    return summarize(ctx, acc, step_count=4, elapsed=1.0)


def test_validation_drawdown_uses_the_registry_risk() -> None:
    """MaxDD compounds the equity at the model's risk per trade."""
    multiples = [2.0, -1.0, -1.0]
    assert _summary(None)["max_drawdown"] == pytest.approx(max_drawdown_pct(multiples, 0.01))
    assert _summary(0.02)["max_drawdown"] == pytest.approx(max_drawdown_pct(multiples, 0.02))
    assert _summary(0.02)["max_drawdown"] > _summary(None)["max_drawdown"]


def test_quality_scheme_id_carries_the_risk() -> None:
    """A changed risk is a different metric definition; legacy tags mean 1 %."""
    assert quality_scheme("forex") == forex_scheme(0.01) == "forex_weighted_targets_v2_risk0.01"
    assert quality_scheme("forex", 0.02) == "forex_weighted_targets_v2_risk0.02"
    assert anchor_is_comparable(FOREX_SCHEME, "forex")
    assert not anchor_is_comparable(FOREX_SCHEME, "forex", 0.02)
    assert not anchor_is_comparable(forex_scheme(0.01), "forex", 0.02)
    assert anchor_is_comparable(forex_scheme(0.02), "forex", 0.02)
    assert quality_scheme("quality", 0.02) == quality_scheme("quality")


def test_model_card_shows_the_risk_next_to_maxdd() -> None:
    """The card labels MaxDD with the risk it was measured at."""
    assert risk_label(0.01) == "1%"
    assert risk_label(0.005) == "0.5%"
    unified = {
        "forex_predictor": {
            "dataset_type": "forex",
            "evaluation": {"risk_per_trade": 0.02},
            "sota_targets": {"max_drawdown": 12.0},
        }
    }
    card = build_model_readme("forex_predictor", unified, 3, {"max_drawdown": 7.5})
    assert "**MaxDD @ 2% risk**: 7.5%" in card
