"""Run options the generated training notebook passes to the trainer.

The Kaggle kernel bundle (GUI ``POST /kaggle/train``, ``lemtrain kaggle launch``, the
``kaggle_cloud_manager`` CLI) builds its notebook with these options; every option is off
by default, so a plain launch runs the trainer exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.cli_args import OVERRIDE_FROZEN_FLAG


@dataclass(frozen=True)
class LaunchOptions:
    """Trainer switches of one launch."""

    override_frozen: bool = False
    """Resume although a frozen run setting changed (``--override-frozen``, logged)."""

    def trainer_flags(self) -> list[str]:
        """Command-line switches appended to the trainer command."""
        return [OVERRIDE_FROZEN_FLAG] if self.override_frozen else []


DEFAULT_LAUNCH = LaunchOptions()
