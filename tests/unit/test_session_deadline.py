"""Kaggle session deadline: settings, the agreed stop check and the GUI resume chain.

Every Kaggle call is a fake; nothing is pushed and nothing runs on Kaggle.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.unit.test_kaggle_kernel_bundle import _FakeKaggle, _RecordingJobs
from training.cloud.session_deadline import (
    DEADLINE_MARKER,
    SESSION_START_ENV,
    SessionDeadline,
    SessionDeadlineConfig,
    load_session_config,
    session_deadline,
)
from training.notebooks.cells.sync import build_push_cell
from training.notebooks.registry import load_registry, resolve_model_metadata
from training.server import job_kaggle_run
from training.training.context import TrainingContext
from training.training.step_progress import deadline_reached


def test_defaults_and_validation() -> None:
    """Defaults are 12 h with a 30 min margin and no auto-resume; bad values are refused."""
    settings = SessionDeadlineConfig.from_config({})
    assert (settings.limit_hours, settings.margin_minutes) == (12.0, 30.0)
    assert settings.auto_resume is False
    with pytest.raises(ValueError):
        SessionDeadlineConfig.from_config({"kaggle_session": {"margin_minutes": 720}})
    with pytest.raises(TypeError):
        SessionDeadlineConfig.from_config({"kaggle_session": {"auto_resume": "yes"}})


def test_deadline_only_on_kaggle_and_from_session_start() -> None:
    """The stop time counts from the exported session start, minus the margin."""
    config = {"kaggle_session": {"limit_hours": 1, "margin_minutes": 10}}
    assert session_deadline(config, "local", {}, lambda: 0.0) is None
    now = {"t": 1000.0}
    deadline = session_deadline(config, "kaggle", {SESSION_START_ENV: "400"}, lambda: now["t"])
    assert deadline is not None
    assert deadline.stop_at == 400 + 3600 - 600
    assert not deadline.expired()
    now["t"] = 3400.0
    assert deadline.expired()


def test_deadline_reached_in_a_single_process() -> None:
    """Without DDP every step checks the deadline."""
    clock = {"t": 0.0}
    ctx = cast(
        TrainingContext,
        SimpleNamespace(
            state=SimpleNamespace(session_deadline=SessionDeadline(10.0, lambda: clock["t"])),
            hardware=SimpleNamespace(device_info=SimpleNamespace(device="cpu")),
        ),
    )
    assert not deadline_reached(ctx, 1)
    clock["t"] = 11.0
    assert deadline_reached(ctx, 7)


def test_push_cell_skips_release_after_deadline_stop() -> None:
    """The notebook's release publish is guarded by the deadline marker."""
    meta = resolve_model_metadata("codeformer", load_registry())
    source = "".join(build_push_cell(meta)["source"])
    compile(source, "push_cell", "exec")
    guard = source.index(DEADLINE_MARKER)
    assert guard < source.index("[PUBLISH]")


class _DeadlineKaggle(_FakeKaggle):
    """Every session completes with a deadline marker in its output."""

    def kernels_output(self, _handle: str, path: str, file_pattern: str) -> None:
        """Write a payload plus the deadline marker."""
        super().kernels_output(_handle, path, file_pattern)
        model_dir = Path(path) / "LemGendaryModels" / "codeformer"
        (model_dir / DEADLINE_MARKER).write_text("{}", encoding="utf-8")


def _run_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_resume: bool) -> Any:
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(
        f"kaggle_session:\n  auto_resume: {str(auto_resume).lower()}\n  max_resumes: 2\n",
        encoding="utf-8",
    )
    assert load_session_config(project).auto_resume is auto_resume
    api = _DeadlineKaggle()
    monkeypatch.setattr(job_kaggle_run, "resolve_kaggle_credentials", lambda: ("u", "KGAT_x"))
    monkeypatch.setattr(job_kaggle_run, "authenticated_kaggle_api", lambda _u, _k: api)
    jobs = _RecordingJobs(tmp_path / "hub")
    jobs.project_root = project
    event = threading.Event()
    monkeypatch.setattr(event, "wait", lambda _s=None: False)
    jobs.execute_kaggle_training("job", "codeformer", {"auto_pull": False}, event)
    return api, jobs


def test_gui_chain_is_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without auto_resume a deadline stop completes the job after one session."""
    api, jobs = _run_chain(tmp_path, monkeypatch, auto_resume=False)
    assert len(api.pushed) == 1
    assert jobs.statuses[-1]["metrics"]["deadline_stop"] is True


def test_gui_chain_resumes_up_to_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With auto_resume the job pushes a new kernel version per deadline stop, up to max."""
    api, jobs = _run_chain(tmp_path, monkeypatch, auto_resume=True)
    assert len(api.pushed) == 3
    assert jobs.statuses[-1]["metrics"]["resumes"] == 2
