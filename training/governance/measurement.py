"""Whether an epoch's metrics are a real measurement (R2-T10).

A Forex ``metrics.csv`` showed an epoch whose metric columns equalled the registry SOTA
targets digit for digit, with ``Quality_Score = 100 - Val_Loss``: the targets had been
written where measurements belong, and that row then anchored the best score and the
per-metric vault baselines of every later epoch. Two guards keep that from happening:

* :func:`unmeasured_reason` says why an epoch's metrics cannot rank as a new best (no
  SOTA target measured, a non-finite quality score, or a Forex epoch without a single
  simulated trade); such an epoch is recorded but never becomes ``best.pth`` or a vault
  checkpoint.
* :func:`is_target_echo` flags a ``metrics.csv`` row whose target columns equal the SOTA
  targets exactly; history readers skip it instead of taking it as a baseline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

from training.governance.metrics import normalize_metric_name
from training.governance.quality_score import finite_number, measured_targets

logger = logging.getLogger("lemtrain.governance.measurement")

FOREX_TRADE_METRICS = ("win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio")
"""Forex metrics that exist only when the model opened at least one trade."""

_ECHO_TOLERANCE = 5e-5
"""Half a unit of the 4-decimal ``metrics.csv`` formatting."""


def unmeasured_reason(
    metrics: Mapping[str, Any], targets: Mapping[str, Any], quality: float, task_type: str
) -> str | None:
    """Why the epoch must not rank as a best, or ``None`` when it is a real measurement."""
    if targets and not measured_targets(targets, metrics):
        return "no SOTA target was measured this epoch"
    if not math.isfinite(quality):
        return "the quality score is not a finite number"
    if task_type == "forex" and not any(
        finite_number(metrics.get(k)) is not None for k in FOREX_TRADE_METRICS
    ):
        return "the model opened no trade (only HOLD predictions), so no trade metric exists"
    return None


def is_target_echo(row: Mapping[str, Any], targets: Mapping[str, Any]) -> bool:
    """Whether every SOTA target column of a ``metrics.csv`` row equals its target.

    Needs at least two targets, so a single coincidence never discards a real row.
    """
    if len(targets) < 2:
        return False
    columns = {normalize_metric_name(str(k)): v for k, v in row.items() if k}
    for key, target in targets.items():
        value = finite_number(columns.get(normalize_metric_name(key)))
        expected = finite_number(target)
        if value is None or expected is None or abs(value - expected) > _ECHO_TOLERANCE:
            return False
    return True


def drop_target_echoes(
    rows: Iterable[Mapping[str, Any]], targets: Mapping[str, Any], source: object
) -> list[Mapping[str, Any]]:
    """``rows`` without the :func:`is_target_echo` rows, each skip logged."""
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        if is_target_echo(row, targets):
            logger.warning(
                "Skipping %s row of epoch %s: its metrics equal the SOTA targets exactly, so "
                "they were not measured; it is no baseline for best or vault checkpoints.",
                source,
                row.get("Epoch", "?"),
            )
            continue
        kept.append(row)
    return kept
