"""Forex validation honesty: the two rows of the owner's metrics.csv (R2-T10).

Epoch 1 of that file held the registry SOTA targets in every metric column and
``Quality_Score = 100 - Val_Loss``. Epoch 2 had ``WinRate == DirAcc`` (no trades),
Sharpe and Sortino pinned at -5 and MaxDD pinned at 100 by clips over a cumulative pip
sum. These tests reproduce both symptoms and pin the fixes.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from training.context_resume import reconcile_metrics_csv
from training.governance.governor import SmartTrainingGovernor
from training.governance.measurement import drop_target_echoes, is_target_echo, unmeasured_reason
from training.governance.quality_score import (
    FOREX_SCHEME,
    anchor_is_comparable,
    csv_row_quality,
    quality_scheme,
    score_metrics,
    target_quality_score,
)
from training.model_card import NOT_MEASURED, measured
from training.telemetry import EpochLosses, EpochRow, TelemetryEngine
from training.training.forex_equity import equity_curve, max_drawdown_pct, risk_multiple
from training.training.val_accumulators import ValAccumulators
from training.training.val_summary import summarize

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = yaml.safe_load((_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8"))
TARGETS: dict[str, float] = dict(_REGISTRY["forex_predictor"]["sota_targets"])

HEADER = (
    "Phase,Fold,Epoch,Train_Loss,Val_Loss,LR,DirAcc,WinRate,ProfitFactor,Sharpe,Sortino,"
    "MaxDD,TP_MAE,SL_MAE,DirEntropy,Quality_Score,Pairs,Data,Temp,Clamp,Cooldown,Batch,"
    "Accumulation,Stress\n"
)
ROW_1 = (
    "1,1,1,0.27142311,0.27090213,0.00099606,58.5000,56.0000,1.6500,1.8500,2.1000,12.0000,"
    "12.5000,12.5000,0.8520,99.7291,16,1.00,0.9039,15.0,0,128,1,0.000000\n"
)
ROW_2 = (
    "1,1,2,0.27279217,0.27168010,0.00099115,28.9778,28.9778,0.1692,-5.0000,-5.0000,100.0000,"
    "3.3377,4.2500,1.0899,0.0000,16,1.00,0.8858,15.0,0,128,1,0.000000\n"
)


def _rows(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    path = tmp_path / "metrics.csv"
    path.write_text(HEADER + ROW_1 + ROW_2, encoding="utf-8")
    keys = [str(name) for name in HEADER.strip().split(",")]
    rows: list[dict[str, str]] = [
        dict(zip(keys, line.strip().split(","), strict=True)) for line in (ROW_1, ROW_2)
    ]
    return path, rows


def _forex_ctx() -> Any:
    return SimpleNamespace(model_name="forex_predictor", models=SimpleNamespace(model=None))


def _forex_pass(
    samples: int, correct: int, trades: list[tuple[float, float, bool]]
) -> dict[str, float]:
    acc = ValAccumulators()
    acc.direction.has_forex_metrics = True
    acc.direction.total_forex_samples = samples
    acc.direction.correct_forex_samples = correct
    acc.direction.total_forex_entropy = 0.9 * samples
    acc.direction.total_trades = len(trades)
    acc.direction.winning_trades = sum(1 for *_, won in trades if won)
    for tp, sl, won in trades:
        acc.trades.trade_returns.append(tp if won else -sl)
        multiple = risk_multiple(tp, sl, won)
        if multiple is not None:
            acc.trades.trade_risk_multiples.append(multiple)
    return summarize(_forex_ctx(), acc, step_count=4, elapsed=1.0)


# --- epoch 1: targets written as measurements --------------------------------------


def test_owner_epoch_1_row_is_a_target_echo(tmp_path: Path) -> None:
    """The epoch-1 row equals the registry targets; epoch 2 does not."""
    _, rows = _rows(tmp_path)
    assert is_target_echo(rows[0], TARGETS)
    assert not is_target_echo(rows[1], TARGETS)
    assert drop_target_echoes(rows, TARGETS, "metrics.csv") == [rows[1]]


def test_owner_epoch_1_row_is_no_resume_baseline(tmp_path: Path) -> None:
    """A resume does not take the echoed targets (or 100 - Val_Loss) as its best."""
    path, rows = _rows(tmp_path)
    governor = SmartTrainingGovernor(
        model_info={"dataset_type": "forex", "sota_targets": TARGETS}, config={}
    )
    governor.best_quality = 0.0
    reconcile_metrics_csv(governor, path, 1.0)
    assert governor.best_quality == pytest.approx(csv_row_quality(rows[1], "forex"))
    assert governor.best_quality < target_quality_score(TARGETS, "forex")


def test_an_epoch_without_measured_targets_cannot_rank() -> None:
    """Metrics without any SOTA target are not a measurement."""
    reason = unmeasured_reason({"val_loss": 0.27, "dir_entropy": 0.85}, TARGETS, 99.7, "forex")
    assert reason is not None and "no SOTA target" in reason
    assert unmeasured_reason({"psnr": 30.0}, {"psnr": 32.0}, math.nan, "restoration")


def test_telemetry_never_writes_unmeasured_metrics_as_values(tmp_path: Path) -> None:
    """Absent metrics and a NaN score are empty cells, not 0 and not targets."""
    engine = TelemetryEngine(str(tmp_path), task_type="forex")
    engine.write_epoch_row(
        EpochRow(
            epoch=1,
            losses=EpochLosses(train_loss=0.27, val_loss=0.27, lr=1e-3),
            metrics={"dir_acc": 40.0, "tp_mae": 3.0, "sl_mae": 4.0, "dir_entropy": 1.0},
            quality_score=math.nan,
            governor_state=None,
            stress=0.0,
        )
    )
    line = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
    columns = dict(zip(HEADER.strip().split(","), line, strict=True))
    assert columns["DirAcc"] == "40.0000"
    for name in ("WinRate", "ProfitFactor", "Sharpe", "Sortino", "MaxDD", "Quality_Score"):
        assert columns[name] == "", name


def test_model_card_never_prints_targets_as_results() -> None:
    """Missing metrics read 'not measured'; zero is a measurement."""
    assert measured({}, "dir_acc") == NOT_MEASURED
    assert measured({"dir_acc": 0.0}, "dir_acc") == "0"
    assert measured({"DirAcc": "57.25"}, "dir_acc", "DirAcc") == "57.25"


# --- epoch 2: substitutes and clips -------------------------------------------------


def test_no_trades_reports_no_trade_metrics() -> None:
    """All-HOLD predictions: no win rate (not DirAcc), no PF/Sharpe/Sortino/MaxDD."""
    results = _forex_pass(samples=100, correct=29, trades=[])
    assert results["dir_acc"] == pytest.approx(29.0)
    for key in ("win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio", "max_drawdown"):
        assert key not in results, key
    assert "quality_score" not in results
    reason = unmeasured_reason(results, TARGETS, score_metrics(results, TARGETS, "forex"), "forex")
    assert reason is not None and "no trade" in reason


def test_losing_run_reports_measured_risk_metrics_without_clips() -> None:
    """Sharpe below -5 and a real drawdown percentage are reported as measured."""
    trades = [(10.0, 20.0, False)] * 30 + [(10.0, 21.0, False)] * 30 + [(10.0, 20.0, True)]
    results = _forex_pass(samples=100, correct=29, trades=trades)
    assert results["win_rate"] == pytest.approx(100.0 / 61.0)
    assert results["win_rate"] != results["dir_acc"]
    assert results["sharpe_ratio"] < -5.0
    expected_dd = (1.0 - 0.99**60) * 100.0
    assert results["max_drawdown"] == pytest.approx(expected_dd)
    assert 0.0 < results["max_drawdown"] < 100.0
    assert unmeasured_reason(results, TARGETS, 1.0, "forex") is None


def test_equity_model_compounds_fixed_fractional_risk() -> None:
    """Equity starts at 1.0, a win adds 1 % * TP/SL, a loss takes 1 %."""
    curve = equity_curve([2.0, -1.0, -1.0])
    assert curve == pytest.approx([1.0, 1.02, 1.02 * 0.99, 1.02 * 0.99 * 0.99])
    assert max_drawdown_pct([2.0, -1.0, -1.0]) == pytest.approx((1 - 0.99**2) * 100.0)
    assert max_drawdown_pct([]) == 0.0
    assert risk_multiple(10.0, 0.0, True) is None


def test_score_bounds_apply_inside_the_score_only() -> None:
    """A Sharpe of -50 scores like -5, an infinite PF like 10; the metrics keep values."""
    base = {"dir_acc": 50.0, "max_drawdown": 10.0}
    clipped = score_metrics({**base, "sharpe_ratio": -5.0, "profit_factor": 10.0}, TARGETS, "forex")
    raw = score_metrics(
        {**base, "sharpe_ratio": -50.0, "profit_factor": math.inf}, TARGETS, "forex"
    )
    assert raw == pytest.approx(clipped)


def test_forex_scores_share_one_scale_with_the_target() -> None:
    """Epoch, CSV rescoring and target use the same score; old forex anchors are dropped."""
    assert quality_scheme("forex") == FOREX_SCHEME
    assert not anchor_is_comparable(None, "forex")
    row_values = {
        "DirAcc": "58.5",
        "WinRate": "56",
        "ProfitFactor": "1.65",
        "Sharpe": "1.85",
        "Sortino": "2.1",
        "MaxDD": "12",
        "TP_MAE": "12.5",
        "SL_MAE": "12.5",
    }
    assert csv_row_quality(row_values, "forex") == pytest.approx(
        target_quality_score(TARGETS, "forex")
    )
