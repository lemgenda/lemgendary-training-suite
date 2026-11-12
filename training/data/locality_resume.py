"""Shard-locality settings of checkpoints written before the run-settings record.

Checkpoints now record ``data.locality`` with the other epoch settings of a run
(``training.run_settings``): a resume finishes an interrupted epoch on the recorded
settings and takes over the configured ones from the next epoch. Older checkpoints carry
the settings under :data:`PAYLOAD_KEY` alone, and the oldest carry none (written before
locality sampling was on by default, so they trained with locality off). This module
reads both for such a resume. It is torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.data.locality import LocalityConfig

PAYLOAD_KEY = "data_locality"
"""Checkpoint key of the ``data.locality`` settings in checkpoints before the record."""

LEGACY_LOCALITY = LocalityConfig(window_shards=0, fraction_block_size=0)
"""Settings of a run whose checkpoint predates :data:`PAYLOAD_KEY` (locality off)."""


def recorded_locality(payload: Mapping[str, Any]) -> LocalityConfig:
    """The settings an old checkpoint was trained with (:data:`LEGACY_LOCALITY` without one).

    Raises:
        TypeError, ValueError: the record is malformed.
    """
    record = payload.get(PAYLOAD_KEY)
    if record is None:
        return LEGACY_LOCALITY
    if not isinstance(record, Mapping):
        raise TypeError(f"checkpoint '{PAYLOAD_KEY}' must be a mapping, got {type(record)}")
    return LocalityConfig.from_section(record)
