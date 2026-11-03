"""Training verdict for the Desktop GUI model catalog.

:func:`training_verdict` is the only place that turns the 4 completion pillars
into ``training_status`` and ``is_fully_trained``. :class:`RungTracker`
replays the ``metrics.csv`` rows as ratchet rungs, so the data-fraction pillar
follows the Monotonic Ratchet pair rule of :mod:`training.ratchet`: a fraction
counts only for the resolution step it was trained at, and it resets when the
resolution steps up.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from training.ratchet import (
    FULL_FRACTION_THRESHOLD,
    RatchetViolation,
    Rung,
    check_transition,
    top_resolution_reached,
)
from training.server.routes.gui_schemas import TrainingStatus

logger = logging.getLogger("lemtrain.server.routes.gui_status")

MIN_FULL_TRAINING_EPOCHS = 20


class TrainingVerdict(NamedTuple):
    """``training_status`` and ``is_fully_trained`` of one model."""

    training_status: TrainingStatus
    is_fully_trained: bool


UNKNOWN_VERDICT = TrainingVerdict("unknown", False)


def training_verdict(
    *,
    epochs_completed: int,
    has_checkpoint: bool,
    sota_reached: bool,
    ladder_passed: bool,
    data_fraction_passed: bool,
) -> TrainingVerdict:
    """Return the training verdict of one model.

    A model is fully trained only when all 4 pillars hold: at least
    ``MIN_FULL_TRAINING_EPOCHS`` epochs, the top ladder rung, full data at that
    rung, and all SOTA targets met.
    """
    is_fully_trained = (
        epochs_completed >= MIN_FULL_TRAINING_EPOCHS
        and sota_reached
        and ladder_passed
        and data_fraction_passed
    )
    status: TrainingStatus
    if is_fully_trained:
        status = "fully_trained"
    elif epochs_completed > 0:
        status = "partially_trained"
    elif has_checkpoint:
        status = "weights_ready"
    else:
        status = "initializing"
    return TrainingVerdict(status, is_fully_trained)


class RungTracker:
    """Replay ``metrics.csv`` rows as ``(resolution, data fraction)`` rungs.

    A row without a resolution keeps the previous one. A row without a data
    fraction keeps the previous fraction only within the same resolution
    step; after a step up the fraction is 0 until a row reports it (pair rule).
    Rows that break the pair rule are logged; the last row still wins, as it
    is what the run last trained at.
    """

    def __init__(self, source: str = "metrics.csv") -> None:
        self._source = source
        self.current: Rung | None = None

    def add(self, resolution: int | None, fraction: float | None) -> None:
        """Record one CSV row."""
        if resolution is None and fraction is None:
            return
        prev = self.current
        if resolution is None:
            resolution = prev.resolution if prev else 0
        if fraction is None:
            fraction = prev.fraction if prev and prev.resolution == resolution else 0.0
        new = Rung(resolution, fraction)
        if prev is not None:
            try:
                check_transition(prev, new)
            except RatchetViolation as exc:
                logger.warning("Ratchet violation in %s: %s", self._source, exc)
        self.current = new

    def full_data_at_top(self, ladder: list[int]) -> bool:
        """True when the last rung is the top ladder rung trained on full data."""
        if self.current is None:
            return False
        return (
            top_resolution_reached(self.current.resolution, ladder)
            and self.current.fraction >= FULL_FRACTION_THRESHOLD
        )
