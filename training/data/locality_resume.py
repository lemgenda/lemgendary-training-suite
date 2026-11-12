"""Shard-locality settings across a resume: a run keeps the settings it started with.

``data.locality`` decides the epoch order (``window_shards``) and, below a data
fraction of 1.0, which samples form the curriculum subset (``fraction_block_size``).
Changing either in the middle of a run would make a mid-epoch resume replay or skip
samples and would swap the subset the run has been training on. Every checkpoint
therefore records the settings in effect (:data:`PAYLOAD_KEY`), and a resume pins the
recorded settings into the run's config before the train dataset and loader are
rebuilt:

* checkpoint records settings equal to the config: nothing changes;
* checkpoint records other settings: the recorded ones win, logged as a warning;
* checkpoint predates the record (written before locality sampling was on by
  default): the run continues with locality off (uniform order, per-sample subset),
  which is what it was trained with, logged as a warning.

New settings take effect for a fresh run (``--clean``). This module is torch-free.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from training.data.locality import LocalityConfig

logger = logging.getLogger("lemtrain.data.locality")

PAYLOAD_KEY = "data_locality"
"""Checkpoint key holding the ``data.locality`` settings the run trains with."""

LEGACY_LOCALITY = LocalityConfig(window_shards=0, fraction_block_size=0)
"""Settings of a run whose checkpoint predates :data:`PAYLOAD_KEY` (locality off)."""


def locality_record(config: Mapping[str, Any] | None) -> dict[str, int]:
    """The checkpoint record of the settings ``config`` trains with."""
    return LocalityConfig.from_config(config).as_dict()


def recorded_locality(payload: Mapping[str, Any]) -> LocalityConfig:
    """The settings a checkpoint was trained with (:data:`LEGACY_LOCALITY` without a record).

    Raises:
        TypeError, ValueError: the record is malformed.
    """
    record = payload.get(PAYLOAD_KEY)
    if record is None:
        return LEGACY_LOCALITY
    if not isinstance(record, Mapping):
        raise TypeError(f"checkpoint '{PAYLOAD_KEY}' must be a mapping, got {type(record)}")
    return LocalityConfig.from_section(record)


def pin_resumed_locality(config: dict[str, Any], payload: Mapping[str, Any]) -> LocalityConfig:
    """Pin the checkpoint's locality settings into ``config`` and return them."""
    configured = LocalityConfig.from_config(config)
    recorded = recorded_locality(payload)
    if PAYLOAD_KEY not in payload:
        logger.warning(
            "[LOCALITY] Checkpoint predates the data.locality record; the resumed run keeps "
            "the uniform order and per-sample subset it was trained with (configured %s is "
            "used by a fresh run).",
            configured.as_dict(),
        )
    elif recorded != configured:
        logger.warning(
            "[LOCALITY] data.locality changed since the run started (%s -> %s); the resumed "
            "run keeps %s so its order and curriculum subset do not change mid-run.",
            recorded.as_dict(),
            configured.as_dict(),
            recorded.as_dict(),
        )
    else:
        logger.info("[LOCALITY] Resumed run keeps data.locality %s.", recorded.as_dict())
    recorded.pin(config)
    return recorded
