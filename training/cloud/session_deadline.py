"""Kaggle session deadline: stop cleanly before the platform kills the session (K-CK-2).

A Kaggle session ends after ``kaggle_session.limit_hours`` (12 h). The training loop asks
:class:`SessionDeadline` whether the deadline (limit minus ``margin_minutes``) has passed;
when it has, the loop saves ``progress.pth`` wherever it is (generator states included),
syncs the ``pytorch/default`` payload to the Hub, writes :data:`DEADLINE_MARKER` into the
model folder and ends with status ``session_deadline``.

The session start is ``LEMGENDARY_SESSION_START`` (seconds since the epoch), which the
notebook's first cell exports; without it the training process start counts (later than the
real session start, so the margin should cover the notebook set-up).

``auto_resume`` (default off) lets the watcher of a GUI-launched kernel push a new kernel
version when a session ended at its deadline, at most ``max_resumes`` times per job. The
config values are the defaults; ``POST /api/training/kaggle/train`` may override both per job
(:meth:`SessionDeadlineConfig.with_chain`).

A launch may also override the session window (``limit_hours``, ``margin_minutes``) for one
kernel, for example a short test session: :class:`SessionWindowOverride` is written into the
generated notebook as :data:`LIMIT_HOURS_ENV` / :data:`MARGIN_MINUTES_ENV`, and
:func:`session_deadline` prefers them to ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("lemtrain.cloud.session_deadline")

SESSION_START_ENV = "LEMGENDARY_SESSION_START"
"""Session start (seconds since the epoch) exported by the notebook's first cell."""
DEADLINE_MARKER = "session_deadline.json"
"""Written into ``LemGendaryModels/<model>`` when a run stopped at its session deadline."""
LIMIT_HOURS_ENV = "LEMGENDARY_SESSION_LIMIT_HOURS"
"""Per-launch ``limit_hours`` override exported by the generated training cell."""
MARGIN_MINUTES_ENV = "LEMGENDARY_SESSION_MARGIN_MINUTES"
"""Per-launch ``margin_minutes`` override exported by the generated training cell."""
CONFIG_SECTION = "kaggle_session"
DEADLINE_STATUS = "session_deadline"
"""``TrainingSummary.status`` / ``stop_reason`` of a run stopped at its deadline."""


@dataclass(frozen=True)
class SessionDeadlineConfig:
    """The ``kaggle_session`` section of ``config.yaml``."""

    limit_hours: float = 12.0
    margin_minutes: float = 30.0
    auto_resume: bool = False
    max_resumes: int = 3

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> SessionDeadlineConfig:
        """Read and validate ``config['kaggle_session']`` (missing keys keep the defaults)."""
        raw = (config or {}).get(CONFIG_SECTION) or {}
        defaults = cls()
        settings = cls(
            limit_hours=float(raw.get("limit_hours", defaults.limit_hours)),
            margin_minutes=float(raw.get("margin_minutes", defaults.margin_minutes)),
            auto_resume=raw.get("auto_resume", defaults.auto_resume),
            max_resumes=raw.get("max_resumes", defaults.max_resumes),
        )
        settings.check_window()
        settings.check_chain()
        return settings

    def check_window(self) -> None:
        """Validate the session window (``limit_hours``, ``margin_minutes``)."""
        if self.limit_hours <= 0 or self.margin_minutes < 0:
            raise ValueError(f"{CONFIG_SECTION}: limit_hours must be > 0, margin_minutes >= 0")
        if self.margin_minutes * 60 >= self.limit_hours * 3600:
            raise ValueError(f"{CONFIG_SECTION}: margin_minutes must be below limit_hours")

    def with_window(self, override: SessionWindowOverride) -> SessionDeadlineConfig:
        """These settings with a per-launch window override (None fields keep the value)."""
        settings = replace(
            self,
            limit_hours=(
                self.limit_hours if override.limit_hours is None else override.limit_hours
            ),
            margin_minutes=(
                self.margin_minutes if override.margin_minutes is None else override.margin_minutes
            ),
        )
        settings.check_window()
        return settings

    def check_chain(self) -> None:
        """Validate the resume-chain settings (``auto_resume``, ``max_resumes``)."""
        if not isinstance(self.auto_resume, bool):
            raise TypeError(f"{CONFIG_SECTION}.auto_resume must be true or false")
        if (
            isinstance(self.max_resumes, bool)
            or not isinstance(self.max_resumes, int)
            or self.max_resumes < 0
        ):
            raise ValueError(f"{CONFIG_SECTION}.max_resumes must be an integer >= 0")

    def with_chain(
        self, auto_resume: bool | None = None, max_resumes: int | None = None
    ) -> SessionDeadlineConfig:
        """These settings with the per-job resume-chain overrides (None keeps the value)."""
        settings = replace(
            self,
            auto_resume=self.auto_resume if auto_resume is None else auto_resume,
            max_resumes=self.max_resumes if max_resumes is None else max_resumes,
        )
        settings.check_chain()
        return settings


def _env_float(environ: Mapping[str, str], name: str) -> float | None:
    """``environ[name]`` as a float (None when unset); a malformed value is an error."""
    raw = environ.get(name, "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc


@dataclass(frozen=True)
class SessionWindowOverride:
    """Per-launch session window; None fields keep ``config.yaml``'s ``kaggle_session``."""

    limit_hours: float | None = None
    margin_minutes: float | None = None

    def __post_init__(self) -> None:
        if self.limit_hours is not None and self.limit_hours <= 0:
            raise ValueError("Session limit_hours override must be > 0")
        if self.margin_minutes is not None and self.margin_minutes < 0:
            raise ValueError("Session margin_minutes override must be >= 0")
        if (
            self.limit_hours is not None
            and self.margin_minutes is not None
            and self.margin_minutes * 60 >= self.limit_hours * 3600
        ):
            raise ValueError("Session margin_minutes override must be below limit_hours")

    @property
    def active(self) -> bool:
        """Whether any field overrides the configuration."""
        return self.limit_hours is not None or self.margin_minutes is not None

    def environment(self) -> dict[str, str]:
        """The environment variables carrying this override into the trainer process."""
        env: dict[str, str] = {}
        if self.limit_hours is not None:
            env[LIMIT_HOURS_ENV] = repr(float(self.limit_hours))
        if self.margin_minutes is not None:
            env[MARGIN_MINUTES_ENV] = repr(float(self.margin_minutes))
        return env

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> SessionWindowOverride:
        """The override exported by the generated notebook (empty when none)."""
        return cls(
            limit_hours=_env_float(environ, LIMIT_HOURS_ENV),
            margin_minutes=_env_float(environ, MARGIN_MINUTES_ENV),
        )


@dataclass(frozen=True)
class SessionDeadline:
    """The moment (``time.time`` scale) after which training must stop."""

    stop_at: float
    clock: Callable[[], float] = time.time

    def expired(self) -> bool:
        """Whether the stop time has passed."""
        return self.clock() >= self.stop_at

    def remaining(self) -> float:
        """Seconds left before the stop time (negative once passed)."""
        return self.stop_at - self.clock()


def session_deadline(
    config: Mapping[str, Any] | None,
    env: str,
    environ: Mapping[str, str] = os.environ,
    clock: Callable[[], float] = time.time,
) -> SessionDeadline | None:
    """The deadline of a Kaggle run (``env == "kaggle"``); None on every other platform."""
    if env != "kaggle":
        return None
    override = SessionWindowOverride.from_environ(environ)
    settings = SessionDeadlineConfig.from_config(config).with_window(override)
    if override.active:
        logger.info("Kaggle session window overridden by the launch: %s", override)
    raw_start = environ.get(SESSION_START_ENV, "")
    try:
        start = float(raw_start) if raw_start else clock()
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using now", SESSION_START_ENV, raw_start)
        start = clock()
    stop_at = start + settings.limit_hours * 3600 - settings.margin_minutes * 60
    logger.info(
        "Kaggle session deadline in %.1f min (limit %.1f h, margin %.0f min)",
        (stop_at - clock()) / 60,
        settings.limit_hours,
        settings.margin_minutes,
    )
    return SessionDeadline(stop_at=stop_at, clock=clock)


class SessionDeadlineReached(RuntimeError):
    """Raised inside an epoch when the session deadline passed (progress already saved)."""

    def __init__(self, epoch: int, phase: str, step: int) -> None:
        super().__init__(f"Kaggle session deadline reached at epoch {epoch} ({phase} {step})")
        self.epoch = epoch
        self.phase = phase
        self.step = step


def write_deadline_marker(model_dir: Path, epoch: int, phase: str, step: int) -> Path:
    """Record in ``model_dir`` that the run stopped at its deadline (read by the watchers)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    marker = model_dir / DEADLINE_MARKER
    marker.write_text(
        json.dumps(
            {"epoch": epoch, "phase": phase, "step": step, "stopped_at": time.time()}, indent=2
        ),
        encoding="utf-8",
    )
    return marker


def load_session_config(project_root: Path) -> SessionDeadlineConfig:
    """``kaggle_session`` of ``<project_root>/config.yaml`` (defaults when absent)."""
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        return SessionDeadlineConfig()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return SessionDeadlineConfig.from_config(raw if isinstance(raw, dict) else {})
