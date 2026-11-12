"""The scalar quality score: one definition for the epoch score and the curriculum target.

Restoration, enhancement and face models use the ``ROBOTS.md`` Scalar Restoration Quality
Score on every epoch::

    Quality Score = PSNR + 20 * SSIM - 20 * LPIPS

FID is measured only every few epochs. It used to enter the weighted score as
``(100 - FID)`` on the epochs that measured it, so the score jumped by 60 to 90 points
every fifth epoch and ``best.pth`` followed the FID cadence instead of the model. FID is
now kept out of the score of these tasks; it still gates publication and triggers
``vault_fid.pth`` from its own column.

Other tasks keep the weighted sum of their SOTA targets (``DEFAULT_METRIC_WEIGHTS``).

Scores written before this change are on another scale. :data:`QUALITY_SCHEME_KEY`
tags governor state and checkpoints with the scheme that produced their scores;
:func:`anchor_is_comparable` says whether a restored best score may be reused, and
:func:`csv_row_quality` rescores old ``metrics.csv`` rows from their metric columns.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS, DEFAULT_METRIC_WEIGHTS
from training.registry.forex_risk import DEFAULT_RISK_PER_TRADE

RESTORATION_QS_TASKS = frozenset({"restoration", "enhancement", "face"})
"""Dataset types scored with the ROBOTS restoration formula."""

QUALITY_SCHEME_KEY = "quality_scheme"
"""Key naming the score scheme in governor state and checkpoint payloads."""

RESTORATION_SCHEME = "robots_psnr_ssim_lpips_v1"
WEIGHTED_SCHEME = "weighted_targets_v1"
FOREX_SCHEME = "forex_weighted_targets_v2"
"""Forex epochs scored by :func:`score_metrics` (R2-T10). Before, ``val_summary`` scored
them with its own ``40*DirAcc + 30*WinRate + ...`` formula on percent values, so the
targets and the epochs were on different scales and the score floored at 0 for any run
whose clipped drawdown reached 100.

The scheme id of a Forex run carries its ``evaluation.risk_per_trade``
(:func:`forex_scheme`, e.g. ``forex_weighted_targets_v2_risk0.01``): the drawdown, and so
the score, depends on it. The bare tag was written before the setting existed, at 1 %."""
FOREX_TASKS = frozenset({"forex"})
_FOREX_CSV_COLUMNS = {
    "dir_acc": "DirAcc",
    "win_rate": "WinRate",
    "profit_factor": "ProfitFactor",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "max_drawdown": "MaxDD",
    "tp_mae": "TP_MAE",
    "sl_mae": "SL_MAE",
}

SSIM_FACTOR = 20.0
LPIPS_FACTOR = 20.0

# metrics.csv column of each restoration-formula metric.
_CSV_COLUMNS = {"psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS"}
_CSV_QUALITY_KEYS = ("Quality", "Quality_Score", "quality_score")

# Lower-is-better metrics: the contribution rises as the value falls.
_LOWER_IS_BETTER_TERM: dict[str, Callable[[float], float]] = {
    "fid": lambda v: 100.0 - v,
    "lpips": lambda v: 1.0 - v,
    "rank_margin": lambda v: 10.0 - v,
    "max_drawdown": lambda v: max(0.0, 100.0 - v),
    "tp_mae": lambda v: max(0.0, 50.0 - v),
    "sl_mae": lambda v: max(0.0, 50.0 - v),
    # Max entropy for 3 classes = ln(3) ~= 1.099: 0 at random, rising with confidence.
    "dir_entropy": lambda v: max(0.0, 1.099 - v),
}


# Bounds applied to a metric inside the weighted score only; the reported metric keeps its
# measured value (an infinite profit factor of a run without losses still scores 10).
_SCORE_BOUNDS: dict[str, tuple[float, float]] = {
    "profit_factor": (0.0, 10.0),
    "sharpe_ratio": (-5.0, 10.0),
    "sortino_ratio": (-5.0, 10.0),
}


def _score_value(key: str, value: Any) -> float | None:
    """The value ``key`` contributes to the score: bounded when listed, else finite only."""
    bounds = _SCORE_BOUNDS.get(key)
    if bounds is None:
        return finite_number(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return min(max(number, bounds[0]), bounds[1])


def uses_restoration_formula(task_type: str) -> bool:
    """Whether ``task_type`` is scored with ``PSNR + 20*SSIM - 20*LPIPS``."""
    return task_type in RESTORATION_QS_TASKS


def forex_scheme(risk: float) -> str:
    """Forex scheme id at ``risk`` per trade (the drawdown depends on it)."""
    return f"{FOREX_SCHEME}_risk{risk:g}"


_LEGACY_SCHEME_ALIASES: dict[str, str] = {FOREX_SCHEME: forex_scheme(DEFAULT_RISK_PER_TRADE)}
"""Scheme tags written before a scheme option existed, mapped to the id they mean."""


def quality_scheme(task_type: str, risk_per_trade: float = DEFAULT_RISK_PER_TRADE) -> str:
    """Name of the score scheme ``task_type`` uses (Forex: at ``risk_per_trade``)."""
    if uses_restoration_formula(task_type):
        return RESTORATION_SCHEME
    return forex_scheme(risk_per_trade) if task_type in FOREX_TASKS else WEIGHTED_SCHEME


def finite_number(value: Any) -> float | None:
    """``value`` as a finite float, else ``None`` (absent, non-numeric, NaN or infinite)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def restoration_quality_score(metrics: Mapping[str, Any]) -> float | None:
    """``PSNR + 20*SSIM - 20*LPIPS`` over the measured terms; ``None`` without PSNR and SSIM."""
    psnr = finite_number(metrics.get("psnr"))
    ssim = finite_number(metrics.get("ssim"))
    if psnr is None and ssim is None:
        return None
    lpips = finite_number(metrics.get("lpips"))
    return (psnr or 0.0) + SSIM_FACTOR * (ssim or 0.0) - LPIPS_FACTOR * (lpips or 0.0)


def weighted_quality_score(
    metrics: Mapping[str, Any], targets: Mapping[str, Any], exclude: frozenset[str] = frozenset()
) -> float:
    """Weighted sum over the SOTA ``targets`` measured in ``metrics`` (absent ones add 0)."""
    score = 0.0
    for key in targets:
        if key in exclude:
            continue
        value = _score_value(key, metrics.get(key))
        if value is None:
            continue
        weight = DEFAULT_METRIC_WEIGHTS.get(key, 1.0)
        if DEFAULT_METRIC_DIRECTIONS.get(key, True):
            score += value * weight
        elif key in _LOWER_IS_BETTER_TERM:
            score += _LOWER_IS_BETTER_TERM[key](value) * weight
        else:
            score += (1.0 / (value + 1e-6)) * weight
    return score


def score_metrics(metrics: Mapping[str, Any], targets: Mapping[str, Any], task_type: str) -> float:
    """The quality score of ``metrics`` (an epoch's readings or the SOTA targets)."""
    if uses_restoration_formula(task_type):
        restoration = restoration_quality_score(metrics)
        if restoration is not None:
            return restoration
        return weighted_quality_score(metrics, targets, exclude=frozenset({"fid"}))
    return weighted_quality_score(metrics, targets)


def measured_targets(
    targets: Mapping[str, Any], measured: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The ``targets`` that ``measured`` holds a finite value for (all when ``None``)."""
    if measured is None:
        return dict(targets)
    return {
        key: value for key, value in targets.items() if finite_number(measured.get(key)) is not None
    }


def target_quality_score(
    targets: Mapping[str, Any], task_type: str, measured: Mapping[str, Any] | None = None
) -> float:
    """The quality score of the SOTA ``targets`` themselves, the curriculum's aim.

    With ``measured`` (an epoch's metrics) only the targets measured that epoch count, so
    the aim is on the same terms as the epoch score it is compared with (R2-T7): FID
    measured every few epochs, or a rank margin that needs two samples per batch, no
    longer raise the aim on the epochs that do not measure them. The same
    :func:`score_metrics` scores both. A non-positive score falls back to ``1.0``; a
    quality-model score at or below 1 is scaled to percent.
    """
    subset = measured_targets(targets, measured)
    target_score = score_metrics(subset, subset, task_type)
    if target_score <= 0:
        return 1.0
    if target_score <= 1.0 and task_type == "quality":
        target_score *= 100.0
    return target_score


def csv_row_quality(row: Mapping[str, str], task_type: str) -> float | None:
    """Quality of one ``metrics.csv`` row on the current scale (``None`` when unknown).

    Restoration-formula rows are rescored from their PSNR/SSIM/LPIPS columns and Forex
    rows from their trade-metric columns, so rows written under an older score still
    compare; other rows keep their stored score.
    """
    if uses_restoration_formula(task_type):
        values = {key: row.get(column) for key, column in _CSV_COLUMNS.items()}
        return restoration_quality_score({k: v for k, v in values.items() if v not in ("", None)})
    if task_type in FOREX_TASKS:
        forex = {key: row.get(column) for key, column in _FOREX_CSV_COLUMNS.items()}
        measured = {k: v for k, v in forex.items() if v not in ("", None)}
        return weighted_quality_score(measured, measured) if measured else None
    for key in _CSV_QUALITY_KEYS:
        value = finite_number(row.get(key))
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ScoreScheme:
    """The task type and the options its quality score scheme depends on."""

    task_type: str = ""
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE

    @property
    def scheme_id(self) -> str:
        """The scheme id (:func:`quality_scheme`)."""
        return quality_scheme(self.task_type, self.risk_per_trade)

    def is_comparable(self, saved_scheme: Any) -> bool:
        """Whether a score saved under ``saved_scheme`` is on this scheme's scale."""
        return anchor_is_comparable(saved_scheme, self.task_type, self.risk_per_trade)


def anchor_is_comparable(
    saved_scheme: Any, task_type: str, risk_per_trade: float = DEFAULT_RISK_PER_TRADE
) -> bool:
    """Whether a best score saved under ``saved_scheme`` is on ``task_type``'s current scale.

    Untagged scores predate the scheme tag; they match only for weighted-score tasks,
    whose formula did not change. A Forex score is comparable only at the same
    ``risk_per_trade``.
    """
    current = quality_scheme(task_type, risk_per_trade)
    if saved_scheme is None:
        return current == WEIGHTED_SCHEME
    return _LEGACY_SCHEME_ALIASES.get(saved_scheme, saved_scheme) == current


def payload_quality_scheme(payload: Mapping[str, Any]) -> Any:
    """Score scheme recorded in a checkpoint payload (its governor state), if any."""
    governor_state = payload.get("governor_state")
    if isinstance(governor_state, Mapping):
        return governor_state.get(QUALITY_SCHEME_KEY)
    return None
