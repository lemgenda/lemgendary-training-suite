"""``POST /api/training/kaggle/train``: per-job auto-resume options over the config defaults.

The job queue is replaced by a recorder, so nothing runs and nothing reaches Kaggle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from training.server.app import create_app

_MODEL = "codeformer"


def _client_and_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> tuple[TestClient, list[dict[str, Any]]]:
    """A test client whose job manager reads ``config`` and records submitted params."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text(config, encoding="utf-8")
    app = create_app(project_root=tmp_path / "state", enforce_auth=False)
    manager = app.state.job_manager
    submitted: list[dict[str, Any]] = []

    def record(job_type: str, model_key: str, params: dict[str, Any]) -> str:
        submitted.append({"job_type": job_type, "model_key": model_key, "params": params})
        return "job-1"

    monkeypatch.setattr(manager, "project_root", project)
    monkeypatch.setattr(manager, "submit_job", record)
    return TestClient(app), submitted


def test_train_defaults_come_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the options the job records the config's kaggle_session values."""
    config = "kaggle_session:\n  auto_resume: true\n  max_resumes: 4\n"
    client, submitted = _client_and_jobs(tmp_path, monkeypatch, config)
    with client:
        resp = client.post("/api/training/kaggle/train", json={"model": _MODEL})
    assert resp.status_code == 200, resp.text
    params = submitted[0]["params"]
    assert (params["auto_resume"], params["max_resumes"]) == (True, 4)


def test_train_request_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit options override the config for this job only."""
    config = "kaggle_session:\n  auto_resume: false\n  max_resumes: 3\n"
    client, submitted = _client_and_jobs(tmp_path, monkeypatch, config)
    payload = {"model": _MODEL, "auto_resume": True, "max_resumes": 1}
    with client:
        resp = client.post("/api/training/kaggle/train", json=payload)
    assert resp.status_code == 200, resp.text
    params = submitted[0]["params"]
    assert (params["auto_resume"], params["max_resumes"]) == (True, 1)


@pytest.mark.parametrize(
    "bad", [{"max_resumes": -1}, {"auto_resume": "maybe"}, {"max_resumes": 1.5}]
)
def test_train_rejects_bad_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: dict[str, Any]
) -> None:
    """Invalid options answer 422 and enqueue nothing."""
    client, submitted = _client_and_jobs(tmp_path, monkeypatch, "")
    with client:
        resp = client.post("/api/training/kaggle/train", json={"model": _MODEL, **bad})
    assert resp.status_code == 422, resp.text
    assert not submitted
