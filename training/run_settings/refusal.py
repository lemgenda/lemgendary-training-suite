"""Resume refusal record: the frozen-settings refusal text, left for the Kaggle job watcher.

A resume refused by :class:`~training.run_settings.policy.FrozenSettingsChanged` ends the
trainer. A local GUI job records the exception text as its error; on Kaggle the trainer runs
inside the kernel, so it also writes that text to :data:`RESUME_REFUSAL_RECORD` in
``LemGendaryModels/<model>`` (saved as kernel output). The Kaggle job reads the record when
the kernel ends and records the same text as its error, so the GUI parses one wording.
Every resume attempt removes a stale record first. This module is torch-free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from training.run_settings.policy import FrozenSettingsChanged

logger = logging.getLogger("lemtrain.run_settings")

RESUME_REFUSAL_RECORD = "resume_refusal.json"
"""File name of the refusal record inside the model's ``LemGendaryModels`` folder."""


@dataclass(frozen=True)
class ResumeRefusalRecord:
    """The refusal record of one run folder."""

    path: Path

    @classmethod
    def in_folder(cls, folder: Path) -> ResumeRefusalRecord:
        """The record of the run folder ``folder`` (``LemGendaryModels/<model>``)."""
        return cls(folder / RESUME_REFUSAL_RECORD)

    def write(self, refused: FrozenSettingsChanged) -> None:
        """Record ``refused``'s text and changed settings; a write failure is logged."""
        payload = {"error": str(refused), "settings": sorted(refused.changes)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as write_err:
            logger.warning("[SETTINGS] Could not write %s: %s", self.path, write_err)

    def clear(self) -> None:
        """Remove a record left by an earlier run; a removal failure is logged."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as remove_err:
            logger.warning("[SETTINGS] Could not remove %s: %s", self.path, remove_err)

    def read(self) -> str | None:
        """The recorded refusal text; ``None`` without a record or with an unreadable one."""
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as read_err:
            logger.warning("[SETTINGS] Unreadable resume refusal %s: %s", self.path, read_err)
            return None
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, str) or not error:
            logger.warning("[SETTINGS] Resume refusal %s has no error text", self.path)
            return None
        return error
