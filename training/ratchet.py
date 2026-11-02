"""Monotonic Ratchet (pair rule) and curriculum rung helpers.

The governor moves through ``(resolution, data fraction)`` rungs. ROBOTS.md
forbids stepping backwards. The owner ruling is the *pair rule*:

* resolution never decreases;
* within one resolution step the data fraction never decreases;
* when the resolution steps up, the fraction may reset to a lower value.

This module is torch-free so the rule can be tested without the training
stack. ``SmartTrainingGovernor`` routes every curriculum transition through
:func:`check_transition`, and both ``get_state()`` and the 4-pillar
completion check use the state keys defined here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

# Canonical governor-state keys. ``SmartTrainingGovernor.get_state()`` writes
# these keys and every reader must use the same constants.
RESOLUTION_STATE_KEY = "input_size"
FRACTION_STATE_KEY = "sample_fraction"

FULL_FRACTION_THRESHOLD = 0.999


class RatchetViolation(ValueError):
    """Raised when a curriculum transition would step backwards."""


class Rung(NamedTuple):
    """One curriculum position: spatial resolution and data fraction."""

    resolution: int
    fraction: float


def check_transition(prev: Rung, new: Rung) -> Rung:
    """Validate ``prev -> new`` under the pair rule and return ``new``.

    Raises:
        RatchetViolation: resolution decreases, or the fraction decreases
            while the resolution stays the same.
    """
    if new.resolution < prev.resolution:
        raise RatchetViolation(
            f"Resolution may not decrease: {prev.resolution}px -> {new.resolution}px."
        )
    if new.resolution == prev.resolution and new.fraction < prev.fraction:
        raise RatchetViolation(
            f"Data fraction may not decrease within the {prev.resolution}px step: "
            f"{prev.fraction:.4f} -> {new.fraction:.4f}."
        )
    return new


def assert_pair_monotonic(trajectory: Iterable[Rung]) -> None:
    """Check every consecutive pair of a trajectory with :func:`check_transition`."""
    previous: Rung | None = None
    for rung in trajectory:
        if previous is not None:
            check_transition(previous, rung)
        previous = rung


def expand_fraction(current: float, increment: float) -> float:
    """Return the next data fraction within the current resolution step.

    Values within 1% of full data snap to 1.0. The result never falls below
    ``current``.
    """
    next_frac = min(1.0, round(current + increment, 4))
    if 1.0 - next_frac < 0.01:
        next_frac = 1.0
    return max(current, next_frac)


def next_rung_resolution(ladder: Sequence[int], current_res: int) -> int | None:
    """Return the next ladder resolution above ``current_res``, or ``None`` at the top."""
    higher = sorted(r for r in ladder if r > current_res)
    return higher[0] if higher else None


def _as_resolution(raw: Any) -> int:
    """Normalize an ``input_size`` value (int or ``[C, H, W]``/``[H, W]``) to pixels."""
    if isinstance(raw, (list, tuple)):
        if not raw:
            return 0
        raw = raw[1] if len(raw) > 1 else raw[0]
    return int(raw or 0)


def executed_rung(state: Mapping[str, Any]) -> Rung:
    """Return the rung recorded in a governor state taken *before* the epoch audit.

    A missing key yields 0 so a state without curriculum data can never pass
    the resolution or data-fraction pillar.
    """
    return Rung(
        resolution=_as_resolution(state.get(RESOLUTION_STATE_KEY, 0)),
        fraction=float(state.get(FRACTION_STATE_KEY, 0.0) or 0.0),
    )


def top_resolution_reached(executed_res: int, ladder: Sequence[int]) -> bool:
    """True when the executed resolution is at or above the top ladder rung."""
    if not ladder:
        return False
    return executed_res >= max(ladder)
