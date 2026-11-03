"""Stochastic Weight Averaging (SWA) settings from the ``swa`` config section.

The section is optional and SWA is disabled when it is missing. This module is
torch-free so the settings can be validated without the training stack; the
averaging itself lives in :mod:`training.training.swa`.

Keys (all optional):

* ``enabled`` (bool, default ``false``): turn SWA on.
* ``start_epoch`` (int >= 1 or null): first epoch whose weights are averaged.
  Overrides ``start_fraction`` when set.
* ``start_fraction`` (0 <= f < 1, default 0.75): averaging starts after this
  fraction of the epoch budget (0.75 averages the last quarter).
* ``swa_lr`` (float > 0 or null): learning rate that ``SWALR`` anneals to in
  the SWA phase. ``null`` keeps the main scheduler running (averaging only).
* ``anneal_epochs`` (int >= 0, default 5) and ``anneal_strategy``
  (``cos`` or ``linear``): the ``SWALR`` annealing.
* ``update_every`` (int >= 1, default 1): average every N epochs of the phase.
* ``eval_every`` (int >= 0, default 0): refresh BatchNorm, validate the
  averaged model and write ``vault_swa.pth`` every N averaging updates.
  ``0`` does this only once, when training ends.
* ``bn_update_batches`` (int >= 1 or null): cap for the BatchNorm refresh
  pass; ``null`` uses the whole train loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CONFIG_SECTION = "swa"
ANNEAL_STRATEGIES = ("cos", "linear")


@dataclass(frozen=True)
class SwaStart:
    """When averaging begins: an explicit epoch, else a fraction of the epoch budget."""

    epoch: int | None = None
    fraction: float = 0.75

    def first_epoch(self, total_epochs: int) -> int:
        """First averaged epoch (1-based) for a budget of ``total_epochs`` epochs."""
        if self.epoch is not None:
            return self.epoch
        return int(max(1, total_epochs) * self.fraction) + 1


@dataclass(frozen=True)
class SwaLrSchedule:
    """``SWALR`` settings: the SWA learning rate and how it is annealed."""

    swa_lr: float
    anneal_epochs: int = 5
    anneal_strategy: str = "cos"


@dataclass(frozen=True)
class SwaConfig:
    """Validated SWA settings (see the module docstring for the keys).

    ``lr`` is ``None`` when ``swa_lr`` is not configured; the main scheduler then
    keeps the learning rate.
    """

    enabled: bool = False
    start: SwaStart = SwaStart()
    lr: SwaLrSchedule | None = None
    update_every: int = 1
    eval_every: int = 0
    bn_update_batches: int | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> SwaConfig:
        """Build the settings from the parsed ``config.yaml``.

        Raises:
            TypeError: the section is not a mapping or a value has the wrong type.
            ValueError: a value is out of range.
        """
        section = (config or {}).get(CONFIG_SECTION) or {}
        if not isinstance(section, Mapping):
            raise TypeError(f"'{CONFIG_SECTION}' must be a mapping, got {type(section).__name__}")
        enabled = section.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError(f"'{CONFIG_SECTION}.enabled' must be a boolean, got {enabled!r}")
        return cls(
            enabled=enabled,
            start=SwaStart(
                epoch=_optional_int(section, "start_epoch", minimum=1),
                fraction=_fraction(section),
            ),
            lr=_lr_schedule(section),
            update_every=_int(section, "update_every", default=1, minimum=1),
            eval_every=_int(section, "eval_every", default=0, minimum=0),
            bn_update_batches=_optional_int(section, "bn_update_batches", minimum=1),
        )

    def first_epoch(self, total_epochs: int) -> int:
        """First averaged epoch (1-based) for a budget of ``total_epochs`` epochs."""
        return self.start.first_epoch(total_epochs)


def _lr_schedule(section: Mapping[str, Any]) -> SwaLrSchedule | None:
    strategy = section.get("anneal_strategy", "cos")
    if strategy not in ANNEAL_STRATEGIES:
        raise ValueError(
            f"'{CONFIG_SECTION}.anneal_strategy' must be one of {ANNEAL_STRATEGIES}, "
            f"got {strategy!r}"
        )
    anneal_epochs = _int(section, "anneal_epochs", default=5, minimum=0)
    swa_lr = _optional_positive_float(section, "swa_lr")
    if swa_lr is None:
        return None
    return SwaLrSchedule(swa_lr=swa_lr, anneal_epochs=anneal_epochs, anneal_strategy=strategy)


def _int(section: Mapping[str, Any], key: str, *, default: int, minimum: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"'{CONFIG_SECTION}.{key}' must be an integer, got {raw!r}")
    if raw < minimum:
        raise ValueError(f"'{CONFIG_SECTION}.{key}' must be >= {minimum}, got {raw}")
    return raw


def _optional_int(section: Mapping[str, Any], key: str, *, minimum: int) -> int | None:
    if section.get(key) is None:
        return None
    return _int(section, key, default=minimum, minimum=minimum)


def _optional_positive_float(section: Mapping[str, Any], key: str) -> float | None:
    raw = section.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"'{CONFIG_SECTION}.{key}' must be a number, got {raw!r}")
    if raw <= 0:
        raise ValueError(f"'{CONFIG_SECTION}.{key}' must be positive, got {raw}")
    return float(raw)


def _fraction(section: Mapping[str, Any]) -> float:
    raw = section.get("start_fraction", 0.75)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"'{CONFIG_SECTION}.start_fraction' must be a number, got {raw!r}")
    if not 0.0 <= raw < 1.0:
        raise ValueError(f"'{CONFIG_SECTION}.start_fraction' must be in [0, 1), got {raw}")
    return float(raw)
