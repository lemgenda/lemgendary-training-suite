"""Options of one Kaggle launch that the generated training notebook carries.

The Kaggle kernel bundle (GUI ``POST /kaggle/train``, ``lemtrain kaggle bundle|launch``, the
``kaggle_cloud_manager`` CLI) builds its notebook with these options. Every option keeps the
previous behaviour by default, so a plain launch clones ``main`` and runs the trainer with
``config.yaml``'s ``kaggle_session`` exactly as before:

* ``override_frozen`` appends the trainer's ``--override-frozen`` switch;
* ``git_refs`` are the refs the clone cell checks out (suite and env-manager);
* ``session`` overrides ``kaggle_session.limit_hours`` / ``margin_minutes`` for this kernel
  (exported to the trainer process by the training cell).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from training.cli_args import OVERRIDE_FROZEN_FLAG
from training.cloud.session_deadline import SessionWindowOverride
from training.notebooks.git_ref import DEFAULT_GIT_REFS, GitRefs


@dataclass(frozen=True)
class LaunchOptions:
    """Trainer switches, clone refs and session window of one launch."""

    override_frozen: bool = False
    """Resume although a frozen run setting changed (``--override-frozen``, logged)."""
    git_refs: GitRefs = DEFAULT_GIT_REFS
    """Refs the clone cell checks out (default ``main`` for both repositories)."""
    session: SessionWindowOverride = field(default_factory=SessionWindowOverride)
    """Session window override (default none: ``config.yaml``'s ``kaggle_session``)."""

    def trainer_flags(self) -> list[str]:
        """Command-line switches appended to the trainer command."""
        return [OVERRIDE_FROZEN_FLAG] if self.override_frozen else []

    def trainer_environment(self) -> dict[str, str]:
        """Environment variables the training cell exports before starting the trainer."""
        return self.session.environment()

    @classmethod
    def from_values(
        cls,
        override_frozen: bool = False,
        git_ref: str | None = None,
        env_manager_ref: str | None = None,
        limit_hours: float | None = None,
        margin_minutes: float | None = None,
    ) -> LaunchOptions:
        """Options from flat option values (CLI flags, API fields); raises ``ValueError``."""
        return cls(
            override_frozen=override_frozen,
            git_refs=GitRefs.of(git_ref, env_manager_ref),
            session=SessionWindowOverride(limit_hours, margin_minutes),
        )

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> LaunchOptions:
        """Options from a job parameter mapping (``POST /kaggle/train`` payload fields)."""
        return cls.from_values(
            override_frozen=bool(params.get("override_frozen", False)),
            git_ref=params.get("git_ref"),
            env_manager_ref=params.get("env_manager_ref"),
            limit_hours=params.get("limit_hours"),
            margin_minutes=params.get("margin_minutes"),
        )


DEFAULT_LAUNCH = LaunchOptions()
