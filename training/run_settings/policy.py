"""Resume policy of the run settings: frozen settings guard, epoch settings switch.

Owner decision (R4-C5): a paused run finishes the current epoch on the old settings and
the next epoch picks up the new ones. :class:`RunSettingsTracker` holds the settings of a
run:

* ``frozen``: fixed for the run. A resume whose configured frozen settings differ from
  the checkpoint's raises :class:`FrozenSettingsChanged`, unless the run starts fresh
  (``--clean``) or ``--override-frozen`` accepts the change (logged loudly and recorded
  in the ledger).
* ``active``: the epoch settings in effect, recorded in every checkpoint. A resume makes
  the recorded ones active, so the resumed epoch replays exactly what it was trained
  with, and keeps the configured ones pending until the next epoch (``apply_from`` is
  the checkpoint epoch + 1: the epoch after an interrupted one, or the first epoch of a
  resume at an epoch boundary).

A checkpoint written before the record has no frozen settings to check (logged); its
epoch settings are taken as the configured ones, except ``data.locality``, which the old
``data_locality`` key (or the locality-off default) supplies. This module is torch-free.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from training.data.locality_resume import recorded_locality
from training.run_settings.keys import LOCALITY
from training.run_settings.ledger import (
    EPOCH_SWITCH,
    LEGACY_CHECKPOINT,
    OVERRIDE_FROZEN,
    SettingsLedger,
    report_changes,
)
from training.run_settings.record import (
    RunSettingsRecord,
    SettingChange,
    describe_changes,
    diff_settings,
)

logger = logging.getLogger("lemtrain.run_settings")


class FrozenSettingsChanged(RuntimeError):
    """The configured frozen settings differ from the ones the run was started with."""

    def __init__(self, changes: Mapping[str, SettingChange]) -> None:
        self.changes = dict(changes)
        super().__init__(
            "Resume refused: frozen run settings changed since the run started: "
            f"{describe_changes(self.changes)}. Start a fresh run with --clean, or accept "
            "the change with --override-frozen."
        )


@dataclass(frozen=True)
class FrozenPolicy:
    """How a resume treats changed frozen settings (``--clean``, ``--override-frozen``)."""

    clean: bool = False
    override: bool = False

    @classmethod
    def from_args(cls, args: Any) -> FrozenPolicy:
        """The policy of a parsed CLI namespace (missing flags are off)."""
        return cls(
            clean=bool(getattr(args, "clean", False)),
            override=bool(getattr(args, "override_frozen", False)),
        )


@dataclass(frozen=True)
class PendingSettings:
    """Configured epoch settings that take effect at epoch ``apply_from``."""

    settings: dict[str, Any]
    apply_from: int
    kind: str


@dataclass(frozen=True)
class DueSettings:
    """A change set due at an epoch boundary."""

    kind: str
    changes: dict[str, SettingChange]


class RunSettingsTracker:
    """Frozen, active and pending run settings of one training run."""

    def __init__(
        self,
        frozen: Mapping[str, Any],
        epoch: Mapping[str, Any],
        ledger: SettingsLedger | None = None,
    ) -> None:
        self.frozen: dict[str, Any] = dict(frozen)
        self.active: dict[str, Any] = dict(epoch)
        self.pending: PendingSettings | None = None
        self.ledger = ledger
        self.legacy_checkpoint = False
        """True after resuming a checkpoint written before the run-settings record."""

    def record(self, epoch_index: int) -> RunSettingsRecord:
        """The record a checkpoint of epoch ``epoch_index`` stores."""
        return RunSettingsRecord(dict(self.frozen), dict(self.active), epoch_index)

    def resume(
        self,
        payload: Mapping[str, Any],
        checkpoint_epoch: int,
        policy: FrozenPolicy,
        mid_epoch: bool,
    ) -> dict[str, Any]:
        """Make the checkpoint's epoch settings active and return them.

        The configured settings become pending from ``checkpoint_epoch + 1``.

        Raises:
            FrozenSettingsChanged: a frozen setting changed and ``policy`` does not allow it.
            RunSettingsError: the checkpoint's record is malformed.
        """
        configured = dict(self.active)
        record = RunSettingsRecord.from_payload(payload)
        if record is None:
            self.legacy_checkpoint = True
            recorded = dict(configured)
            recorded[LOCALITY] = recorded_locality(payload).as_dict()
            kind = LEGACY_CHECKPOINT
            logger.warning(
                "[SETTINGS] Checkpoint of epoch %d predates the run-settings record: frozen "
                "settings cannot be checked, the configured epoch settings are taken over "
                "from epoch %d (data.locality %s until then).",
                checkpoint_epoch,
                checkpoint_epoch + 1,
                recorded[LOCALITY],
            )
        else:
            self._guard_frozen(record, policy, checkpoint_epoch)
            recorded = {key: record.epoch.get(key, value) for key, value in configured.items()}
            kind = EPOCH_SWITCH
        self.active = recorded
        changes = diff_settings(recorded, configured)
        if changes:
            self.pending = PendingSettings(configured, checkpoint_epoch + 1, kind)
            logger.info(
                "[SETTINGS] Resume %s epoch %d keeps the recorded settings; from epoch %d the "
                "configured ones apply: %s",
                "inside" if mid_epoch else "after",
                checkpoint_epoch,
                checkpoint_epoch + 1,
                describe_changes(changes),
            )
        return dict(recorded)

    def _guard_frozen(
        self, record: RunSettingsRecord, policy: FrozenPolicy, checkpoint_epoch: int
    ) -> None:
        """Refuse a frozen-settings change unless ``policy`` allows it (then record it)."""
        if policy.clean:
            return
        changes = diff_settings(record.frozen, self.frozen)
        if not changes:
            return
        if not policy.override:
            raise FrozenSettingsChanged(changes)
        logger.warning(
            "[SETTINGS] --override-frozen: resuming with CHANGED FROZEN SETTINGS; checkpoints, "
            "metrics and the best score before epoch %d were made under the old ones.",
            checkpoint_epoch + 1,
        )
        report_changes(self.ledger, checkpoint_epoch, OVERRIDE_FROZEN, changes)

    def due(self, epoch: int) -> DueSettings | None:
        """The pending change set when epoch ``epoch`` starts on or after its turn."""
        pending = self.pending
        if pending is None or epoch < pending.apply_from:
            return None
        self.pending = None
        return DueSettings(pending.kind, diff_settings(self.active, pending.settings))

    def activate(self, settings: Mapping[str, Any]) -> None:
        """Make ``settings`` (a subset of the epoch settings) active."""
        self.active.update(settings)
