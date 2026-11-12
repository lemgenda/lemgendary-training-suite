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
from typing import Any

from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS, DEFAULT_METRIC_WEIGHTS

RESTORATION_QS_TASKS = frozenset({"restoration", "enhancement", "face"})
"""Dataset types scored with the ROBOTS restoration formula."""

QUALITY_SCHEME_KEY = "quality_scheme"
"""Key naming the score scheme in governor state and checkpoint payloads."""

RESTORATION_SCHEME = "robots_psnr_ssim_lpips_v1"
WEIGHTED_SCHEME = "weighted_targets_v1"

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


def uses_restoration_formula(task_type: str) -> bool:
    """Whether ``task_type`` is scored with ``PSNR + 20*SSIM - 20*LPIPS``."""
    return task_type in RESTORATION_QS_TASKS


def quality_scheme(task_type: str) -> str:
    """Name of the score scheme ``task_type`` uses."""
    return RESTORATION_SCHEME if uses_restoration_formula(task_type) else WEIGHTED_SCHEME


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def restoration_quality_score(metrics: Mapping[str, Any]) -> float | None:
    """``PSNR + 20*SSIM - 20*LPIPS`` over the measured terms; ``None`` without PSNR and SSIM."""
    psnr = _finite(metrics.get("psnr"))
    ssim = _finite(metrics.get("ssim"))
    if psnr is None and ssim is None:
        return None
    lpips = _finite(metrics.get("lpips"))
    return (psnr or 0.0) + SSIM_FACTOR * (ssim or 0.0) - LPIPS_FACTOR * (lpips or 0.0)


def weighted_quality_score(
    metrics: Mapping[str, Any], targets: Mapping[str, Any], exclude: frozenset[str] = frozenset()
) -> float:
    """Weighted sum over the SOTA ``targets`` measured in ``metrics`` (absent ones add 0)."""
    score = 0.0
    for key in targets:
        if key in exclude:
            continue
        value = _finite(metrics.get(key))
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


def csv_row_quality(row: Mapping[str, str], task_type: str) -> float | None:
    """Quality of one ``metrics.csv`` row on the current scale (``None`` when unknown).

    Restoration-formula rows are rescored from their PSNR/SSIM/LPIPS columns, so rows
    written under the old weighted score still compare; other rows keep their stored
    score.
    """
    if uses_restoration_formula(task_type):
        values = {key: row.get(column) for key, column in _CSV_COLUMNS.items()}
        return restoration_quality_score({k: v for k, v in values.items() if v not in ("", None)})
    for key in _CSV_QUALITY_KEYS:
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def anchor_is_comparable(saved_scheme: Any, task_type: str) -> bool:
    """Whether a best score saved under ``saved_scheme`` is on ``task_type``'s current scale.

    Untagged scores predate the scheme tag; they match only for weighted-score tasks,
    whose formula did not change.
    """
    current = quality_scheme(task_type)
    if saved_scheme is None:
        return current == WEIGHTED_SCHEME
    return saved_scheme == current


def payload_quality_scheme(payload: Mapping[str, Any]) -> Any:
    """Score scheme recorded in a checkpoint payload (its governor state), if any."""
    governor_state = payload.get("governor_state")
    if isinstance(governor_state, Mapping):
        return governor_state.get(QUALITY_SCHEME_KEY)
    return None
