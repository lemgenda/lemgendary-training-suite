"""Equity model of the Forex validation trade simulation (R2-T10).

Each directional (non-HOLD) validation prediction is one simulated trade that closes
at its labelled take-profit (win) or stop-loss (loss). The equity model is explicit:

* fixed-fractional risk: every trade risks the fraction ``risk`` of the current equity at
  its stop-loss, the common retail money-management rule. ``risk`` is the per-model
  registry setting ``evaluation.risk_per_trade``
  (:func:`training.registry.forex_risk.risk_per_trade`, default
  1 %, at most 5 %);
* a trade's outcome in risk units (R multiple) is ``TP / SL`` for a win and ``-1`` for
  a loss, so its equity return is ``risk * R``;
* equity starts at 1.0 and compounds: ``E_i = E_{i-1} * (1 + risk * R_i)``.

The drawdown scales roughly linearly with ``risk``, so the value is part of the metric
definition: it is fixed for a run (recorded in the checkpoint run settings and in the
Forex quality scheme id) and shown next to MaxDD in the model card.

The maximum drawdown is the largest peak-to-trough fall of that curve as a percentage
of the running peak, ``max_i (P_i - E_i) / P_i * 100`` with ``P_i = max(1, E_0..E_i)``.
It lies in ``[0, 100)`` by construction, so it needs no clipping. (The earlier value
was a drawdown of the cumulative pip sum, clipped to 100: not a percentage of
anything, and pinned at 100 for any losing run.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from training.registry.forex_risk import DEFAULT_RISK_PER_TRADE


def risk_multiple(take_profit: float, stop_loss: float, won: bool) -> float | None:
    """The trade's outcome in units of its risk; ``None`` without a positive stop-loss."""
    if not stop_loss > 0.0:
        return None
    return float(take_profit) / float(stop_loss) if won else -1.0


def equity_curve(r_multiples: Sequence[float], risk: float = DEFAULT_RISK_PER_TRADE) -> np.ndarray:
    """Compounded equity after each trade, starting from 1.0 (the start is element 0)."""
    returns = 1.0 + risk * np.asarray(r_multiples, dtype=np.float64)
    return np.concatenate(([1.0], np.cumprod(np.maximum(returns, 0.0))))


def max_drawdown_pct(r_multiples: Sequence[float], risk: float = DEFAULT_RISK_PER_TRADE) -> float:
    """Largest peak-to-trough fall of the equity curve, in percent of the peak."""
    curve = equity_curve(r_multiples, risk)
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / peaks) * 100.0)


@dataclass(frozen=True)
class RegimeMetrics:
    """Regime-aware risk metrics partitioned by market environment."""

    regime_name: str
    sample_count: int
    win_rate_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float


def _calculate_single_regime(sub_mults: np.ndarray, reg: str, risk: float) -> RegimeMetrics | None:
    count = len(sub_mults)
    if count == 0:
        return None

    wins = int(np.sum(sub_mults > 0))
    win_rate = (float(wins) / float(count)) * 100.0

    gross_profit = float(np.sum(sub_mults[sub_mults > 0]))
    gross_loss = float(np.sum(np.abs(sub_mults[sub_mults < 0])))
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = float("inf") if gross_profit > 0 else 0.0

    mean_r = float(np.mean(sub_mults))
    std_r = float(np.std(sub_mults))
    horizon = np.sqrt(min(252, count))
    sharpe = (mean_r / max(1e-6, std_r)) * horizon
    mdd = max_drawdown_pct(sub_mults, risk)

    return RegimeMetrics(
        regime_name=reg,
        sample_count=count,
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(pf, 4),
        sharpe_ratio=round(float(sharpe), 4),
        max_drawdown_pct=round(mdd, 2),
    )


def compute_regime_metrics(
    r_multiples: Sequence[float],
    regimes: Sequence[str],
    risk: float = DEFAULT_RISK_PER_TRADE,
) -> dict[str, RegimeMetrics]:
    """Calculate Sharpe, WinRate, and MaxDD segmented across regimes (e.g. Bull, Bear, Volatile)."""
    if len(r_multiples) != len(regimes):
        raise ValueError("r_multiples and regimes sequences must have equal length.")

    mult_arr = np.asarray(r_multiples, dtype=np.float64)
    reg_arr = np.asarray(regimes)
    results: dict[str, RegimeMetrics] = {}

    for reg in sorted(set(regimes)):
        metrics = _calculate_single_regime(mult_arr[reg_arr == reg], reg, risk)
        if metrics is not None:
            results[reg] = metrics

    return results
