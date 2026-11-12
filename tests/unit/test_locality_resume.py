"""A resumed run keeps the shard-locality settings it started with (R2-T2)."""

from __future__ import annotations

import logging
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
    locality_record,
    pin_resumed_locality,
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


def test_record_round_trips() -> None:
    """The checkpoint record rebuilds the same settings."""
    record = locality_record(_config(2, 64))
    assert recorded_locality({PAYLOAD_KEY: record}) == LocalityConfig(2, 64)


def test_legacy_checkpoint_resumes_with_locality_off(caplog: pytest.LogCaptureFixture) -> None:
    """A checkpoint without a record keeps the uniform order and per-sample subset."""
    config = _config(4, 256)
    with caplog.at_level(logging.WARNING, logger="lemtrain.data.locality"):
        pinned = pin_resumed_locality(config, {"epoch": 3})
    assert pinned == LEGACY_LOCALITY
    assert LocalityConfig.from_config(config) == LEGACY_LOCALITY
    assert "predates" in caplog.text


def test_changed_settings_keep_the_recorded_ones(caplog: pytest.LogCaptureFixture) -> None:
    """Settings changed mid-run are logged and the recorded ones win."""
    config = _config(4, 256)
    with caplog.at_level(logging.WARNING, logger="lemtrain.data.locality"):
        pinned = pin_resumed_locality(config, {PAYLOAD_KEY: locality_record(_config(8, 32))})
    assert pinned == LocalityConfig(8, 32)
    assert config["data"]["locality"] == {"window_shards": 8, "fraction_block_size": 32}
    assert "changed since the run started" in caplog.text


def test_unchanged_settings_resume_silently(caplog: pytest.LogCaptureFixture) -> None:
    """Matching settings are kept without a warning."""
    config = _config(4, 256)
    with caplog.at_level(logging.WARNING, logger="lemtrain.data.locality"):
        pinned = pin_resumed_locality(config, {PAYLOAD_KEY: locality_record(config)})
    assert pinned == ON
    assert caplog.text == ""


def test_pin_creates_a_missing_section() -> None:
    """Pinning into a config without a data section adds one."""
    config: dict[str, Any] = {}
    LocalityConfig(1, 2).pin(config)
    assert LocalityConfig.from_config(config) == LocalityConfig(1, 2)


def test_malformed_record_is_rejected() -> None:
    """A non-mapping record raises instead of silently resetting the order."""
    with pytest.raises(TypeError):
        recorded_locality({PAYLOAD_KEY: [4, 256]})
