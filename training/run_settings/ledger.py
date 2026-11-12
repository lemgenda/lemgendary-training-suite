"""Log and record every run-settings change (``settings_changes.jsonl`` in the run folder).

Each change set is one ``[SETTINGS] epoch N: ...`` log line and one JSON line in
``LemGendaryModels/<model>/settings_changes.jsonl``::

    {"changes": {"data.locality": {"from": {...}, "to": {...}}}, "epoch": 4,
     "kind": "epoch_switch", "refused": {}, "time": "2026-11-13T10:00:00+00:00"}

``kind`` is ``epoch_switch`` (configured settings taken over at an epoch boundary),
``legacy_checkpoint`` (the same, resuming a checkpoint written before the record) or
``override_frozen`` (a frozen setting changed under ``--override-frozen``). ``refused``
holds the changes that were not applied (e.g. a batch size that does not divide the
effective batch). This module is torch-free.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from training.run_settings.record import SettingChange, canonical_json, describe_changes

logger = logging.getLogger("lemtrain.run_settings")

SETTINGS_LEDGER_NAME = "settings_changes.jsonl"
"""File name of the change ledger inside the model's ``LemGendaryModels`` folder."""

EPOCH_SWITCH = "epoch_switch"
LEGACY_CHECKPOINT = "legacy_checkpoint"
OVERRIDE_FROZEN = "override_frozen"


@dataclass(frozen=True)
class SettingsLedger:
    """Append-only JSON-lines file of run-settings changes."""

    path: Path

    @classmethod
    def in_folder(cls, folder: Path) -> SettingsLedger:
        """The ledger of the run folder ``folder`` (``LemGendaryModels/<model>``)."""
        return cls(folder / SETTINGS_LEDGER_NAME)

    def append(self, entry: Mapping[str, object]) -> None:
        """Append ``entry`` as one JSON line; a write failure is logged, training goes on."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(entry) + "\n")
        except OSError as write_err:
            logger.warning("[SETTINGS] Could not append to %s: %s", self.path, write_err)


def report_changes(
    ledger: SettingsLedger | None,
    epoch: int,
    kind: str,
    applied: Mapping[str, SettingChange],
    refused: Mapping[str, SettingChange] | None = None,
) -> None:
    """Log one change set and append it to ``ledger`` (nothing when both sets are empty)."""
    refused = refused or {}
    if not applied and not refused:
        return
    if applied:
        level = logging.WARNING if kind == OVERRIDE_FROZEN else logging.INFO
        logger.log(level, "[SETTINGS] epoch %d: %s: %s", epoch, kind, describe_changes(applied))
    if refused:
        logger.warning(
            "[SETTINGS] epoch %d: refused, recorded values kept: %s",
            epoch,
            describe_changes(refused),
        )
    if ledger is None:
        return
    ledger.append(
        {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "epoch": epoch,
            "kind": kind,
            "changes": {key: change.as_dict() for key, change in applied.items()},
            "refused": {key: change.as_dict() for key, change in refused.items()},
        }
    )
