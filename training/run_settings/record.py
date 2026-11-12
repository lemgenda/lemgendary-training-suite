"""The run-settings record a checkpoint carries (R4-C5).

Every checkpoint stores :data:`PAYLOAD_KEY` with two groups:

* ``frozen``: settings fixed for the whole run (architecture, optimizer type and
  param-group layout, dataset identity and split, metric definitions, seed, precision),
  plus ``frozen_sha``, the SHA-256 of their canonical JSON;
* ``epoch``: the settings the recorded epoch trained with (data locality, batch size and
  accumulation, LR schedule, weight decay, gradient clipping, loader workers). A resume
  finishes an interrupted epoch on these and switches to the configured ones next epoch.

This module is torch-free: canonical JSON, the hash, the record and the diff of two
settings groups.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PAYLOAD_KEY = "run_settings"
"""Checkpoint key holding the :class:`RunSettingsRecord`."""
FROZEN_KEY = "frozen"
FROZEN_SHA_KEY = "frozen_sha"
EPOCH_KEY = "epoch"
EPOCH_INDEX_KEY = "epoch_index"


class RunSettingsError(ValueError):
    """A checkpoint's run-settings record is malformed."""


def jsonable(value: Any) -> Any:
    """``value`` as plain JSON types: tuples become lists, mapping keys strings."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Key-sorted, whitespace-free JSON of ``value`` (the input of :func:`settings_sha`)."""
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def settings_sha(settings: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON of ``settings``."""
    return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SettingChange:
    """One setting whose value differs between two settings groups."""

    old: Any
    new: Any

    def as_dict(self) -> dict[str, Any]:
        """The change as a ``{"from": ..., "to": ...}`` JSON object."""
        return {"from": jsonable(self.old), "to": jsonable(self.new)}


def diff_settings(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, SettingChange]:
    """The settings whose canonical JSON differs, in key order (missing keys are ``None``)."""
    changes: dict[str, SettingChange] = {}
    for key in sorted(set(old) | set(new)):
        before, after = old.get(key), new.get(key)
        if canonical_json(before) != canonical_json(after):
            changes[key] = SettingChange(before, after)
    return changes


def describe_changes(changes: Mapping[str, SettingChange]) -> str:
    """One-line ``key old -> new; ...`` text of ``changes`` for logs and exceptions."""
    return "; ".join(
        f"{key} {canonical_json(change.old)} -> {canonical_json(change.new)}"
        for key, change in changes.items()
    )


@dataclass(frozen=True)
class RunSettingsRecord:
    """The frozen and epoch settings of a run at a checkpoint."""

    frozen: dict[str, Any]
    epoch: dict[str, Any]
    epoch_index: int = 0

    @property
    def frozen_sha(self) -> str:
        """SHA-256 of the canonical JSON of the frozen settings."""
        return settings_sha(self.frozen)

    def to_payload(self) -> dict[str, Any]:
        """The checkpoint value of :data:`PAYLOAD_KEY`."""
        return {
            FROZEN_KEY: jsonable(self.frozen),
            FROZEN_SHA_KEY: self.frozen_sha,
            EPOCH_KEY: jsonable(self.epoch),
            EPOCH_INDEX_KEY: int(self.epoch_index),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RunSettingsRecord | None:
        """The record of a checkpoint payload; ``None`` for a checkpoint written before it.

        Raises:
            RunSettingsError: the record is not a mapping, misses a group, or its
                ``frozen_sha`` does not match the frozen settings.
        """
        raw = payload.get(PAYLOAD_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise RunSettingsError(f"checkpoint '{PAYLOAD_KEY}' must be a mapping, got {raw!r}")
        frozen, epoch = raw.get(FROZEN_KEY), raw.get(EPOCH_KEY)
        if not isinstance(frozen, Mapping) or not isinstance(epoch, Mapping):
            raise RunSettingsError(
                f"checkpoint '{PAYLOAD_KEY}' needs '{FROZEN_KEY}' and '{EPOCH_KEY}' mappings"
            )
        record = cls(dict(frozen), dict(epoch), int(raw.get(EPOCH_INDEX_KEY, 0) or 0))
        if raw.get(FROZEN_SHA_KEY) != record.frozen_sha:
            raise RunSettingsError(
                f"checkpoint '{PAYLOAD_KEY}.{FROZEN_SHA_KEY}' does not match its frozen settings"
            )
        return record
