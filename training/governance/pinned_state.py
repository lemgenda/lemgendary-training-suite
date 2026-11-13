"""Governor state whose calibration is fixed for the run (quality / NIMA models, R4-C4).

NIMA trains its 10-bin score distribution with a plain softmax (temperature 1) under a tanh
logit soft cap. A temperature in the training loss is redundant (the last linear layer learns
any logit scale) and a time-varying one makes the objective non-stationary; with the cap it
would also change the effective cap. The cap changes the function being trained, so it stays
at the registry ``stabilizers.logit_clamp`` (15) for the whole run. Temperature scaling is a
post-hoc calibration step (``training.quality_calibration``).

The governor's thermal logic (sharpening, recoil, resets, turbulence breakouts, checkpoint
restore) keeps running for every task; for these models the pinned fields simply keep their
start value, so no caller needs a task-type branch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from training.governance.governor_groups import Calibration
from training.governance.thermal import ThermalState

logger = logging.getLogger("lemtrain.governance.pinned_state")


def _set_unless_pinned(obj: Any, name: str, value: Any, pinned: frozenset[str]) -> bool:
    """Whether ``obj.name = value`` may proceed: a pinned field keeps its first value."""
    if name not in pinned or name not in obj.__dict__:
        return True
    current = obj.__dict__[name]
    if value != current:
        logger.debug(
            "%s.%s stays %r for the run (requested %r)", type(obj).__name__, name, current, value
        )
    return False


@dataclass
class PinnedThermalState(ThermalState):
    """Thermal state with the softmax temperature fixed (temperature 1 for NIMA)."""

    pinned_fields: ClassVar[frozenset[str]] = frozenset({"temperature"})

    def __setattr__(self, name: str, value: Any) -> None:
        if _set_unless_pinned(self, name, value, self.pinned_fields):
            super().__setattr__(name, value)


@dataclass
class PinnedCalibration(Calibration):
    """Calibration with the logit clamp (the model's soft cap) fixed for the run."""

    pinned_fields: ClassVar[frozenset[str]] = frozenset({"current_clamp"})

    def __setattr__(self, name: str, value: Any) -> None:
        if _set_unless_pinned(self, name, value, self.pinned_fields):
            super().__setattr__(name, value)
