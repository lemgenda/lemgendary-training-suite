"""Shard-locality settings of checkpoints written before the run-settings record (R2-T2).

The resume behaviour (finish the epoch on the recorded settings, switch next epoch) is
covered in ``test_run_settings.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from training.data.locality import (
    DEFAULT_FRACTION_BLOCK_SIZE,
    DEFAULT_WINDOW_SHARDS,
    LocalityConfig,
)
from training.data.locality_resume import (
    LEGACY_LOCALITY,
    PAYLOAD_KEY,
    recorded_locality,
)

ON = LocalityConfig(DEFAULT_WINDOW_SHARDS, DEFAULT_FRACTION_BLOCK_SIZE)


def _config(window: int, block: int) -> dict[str, Any]:
    return {"data": {"locality": {"window_shards": window, "fraction_block_size": block}}}


def test_workspace_config_turns_locality_on_with_the_module_defaults() -> None:
    """config.yaml and the module defaults agree, and both are on."""
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert LocalityConfig.from_config(config) == ON
    assert ON.window_shards == 4
    assert ON.fraction_block_size > 0


def test_old_record_round_trips() -> None:
    """The old ``data_locality`` record rebuilds the same settings."""
    record = LocalityConfig.from_config(_config(2, 64)).as_dict()
    assert recorded_locality({PAYLOAD_KEY: record}) == LocalityConfig(2, 64)


def test_checkpoint_without_a_record_trained_with_locality_off() -> None:
    """A checkpoint without any record trained with the uniform order and per-sample subset."""
    assert recorded_locality({"epoch": 3}) == LEGACY_LOCALITY
    assert LEGACY_LOCALITY == LocalityConfig(0, 0)


def test_pin_creates_a_missing_section() -> None:
    """Pinning into a config without a data section adds one."""
    config: dict[str, Any] = {}
    LocalityConfig(1, 2).pin(config)
    assert LocalityConfig.from_config(config) == LocalityConfig(1, 2)


def test_malformed_record_is_rejected() -> None:
    """A non-mapping record raises instead of silently resetting the order."""
    with pytest.raises(TypeError):
        recorded_locality({PAYLOAD_KEY: [4, 256]})
