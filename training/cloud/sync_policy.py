"""When the per-epoch Kaggle auto-sync uploads a new ``pytorch/default`` version.

Every upload creates a Kaggle model version, so the auto-sync is throttled by two settings in
the ``fleet`` section of ``config.yaml``:

* ``kaggle_sync_every_epochs`` (default 5): upload on every N-th epoch (0 disables the
  cadence, leaving only new-best uploads);
* ``kaggle_sync_on_best`` (default true): also upload an epoch that set a new best score.

The final sync at the end of a run is not throttled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EVERY_EPOCHS_KEY = "kaggle_sync_every_epochs"
ON_BEST_KEY = "kaggle_sync_on_best"
DEFAULT_EVERY_EPOCHS = 5
DEFAULT_ON_BEST = True


@dataclass(frozen=True)
class KaggleSyncPolicy:
    """Throttle of the per-epoch Kaggle auto-sync."""

    every_epochs: int = DEFAULT_EVERY_EPOCHS
    on_best: bool = DEFAULT_ON_BEST

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> KaggleSyncPolicy:
        """Read the policy from ``config['fleet']``.

        Raises:
            TypeError: ``kaggle_sync_every_epochs`` is not an integer or
                ``kaggle_sync_on_best`` is not a boolean.
            ValueError: ``kaggle_sync_every_epochs`` is negative.
        """
        fleet = (config or {}).get("fleet") or {}
        every = fleet.get(EVERY_EPOCHS_KEY, DEFAULT_EVERY_EPOCHS)
        on_best = fleet.get(ON_BEST_KEY, DEFAULT_ON_BEST)
        if isinstance(every, bool) or not isinstance(every, int):
            raise TypeError(f"fleet.{EVERY_EPOCHS_KEY} must be an integer, got {every!r}")
        if every < 0:
            raise ValueError(f"fleet.{EVERY_EPOCHS_KEY} must be >= 0, got {every}")
        if not isinstance(on_best, bool):
            raise TypeError(f"fleet.{ON_BEST_KEY} must be true or false, got {on_best!r}")
        return cls(every_epochs=every, on_best=on_best)

    def should_sync(self, epoch: int, is_best: bool) -> bool:
        """Whether the epoch that just finished uploads a new version."""
        if self.on_best and is_best:
            return True
        return self.every_epochs > 0 and epoch > 0 and epoch % self.every_epochs == 0
