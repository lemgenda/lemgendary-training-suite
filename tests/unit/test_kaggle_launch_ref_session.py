"""Kaggle launch options: clone git refs and the per-kernel session window.

``--git-ref`` / ``--env-manager-ref`` choose what the generated clone cell checks out
(default ``main``); ``--limit-hours`` / ``--margin-minutes`` are exported by the generated
training cell and preferred by the trainer's session deadline over ``config.yaml``. The
options reach the notebook through ``BundleRequest``, ``lemtrain kaggle bundle|launch``,
``kaggle_cloud_manager`` and ``POST /kaggle/train``. Nothing is pushed: the Kaggle API is a
fake, and the clone helper runs against a local git repository.
"""

from __future__ import annotations

import ast
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.unit.test_kaggle_train_route import _client_and_jobs
from tests.unit.test_override_frozen_launch import run_gui_job
from training import kaggle_cloud_manager
from training.cli import kaggle_commands
from training.cloud.kernel_bundle import BundleRequest, build_kernel_bundle
from training.cloud.session_deadline import (
    LIMIT_HOURS_ENV,
    MARGIN_MINUTES_ENV,
    SESSION_START_ENV,
    SessionWindowOverride,
    session_deadline,
)
from training.notebooks.cells.repo import build_clone_cell
from training.notebooks.cells.sync import build_training_cell
from training.notebooks.git_ref import GitRefError, GitRefs, validate_git_ref
from training.notebooks.launch_options import LaunchOptions
from training.notebooks.registry import load_registry, resolve_model_metadata

_REGISTRY = load_registry()
_MODEL = "upn_v2"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_GIT = shutil.which("git")


def _notebook_text(folder: Path) -> str:
    """Source of every code cell of the one notebook in ``folder``."""
    notebook = json.loads(next(folder.glob("*.ipynb")).read_text(encoding="utf-8"))
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


def _cell_text(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def _refs_in(text: str) -> tuple[str, str]:
    """The ``suite_ref`` and ``env_mgr_ref`` literals of a generated clone cell."""
    values = {
        name.strip(): ast.literal_eval(value.strip())
        for name, value in (
            line.split("=", 1)
            for line in text.splitlines()
            if line.startswith(("suite_ref =", "env_mgr_ref ="))
        )
    }
    return values["suite_ref"], values["env_mgr_ref"]


# --- ref validation ----------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["main", "fix/review-remediation", "v16.9.210", "rel_1-2", _SHA])
def test_valid_refs(ref: str) -> None:
    """Branch, tag and full SHA names pass unchanged."""
    assert validate_git_ref(ref) == ref


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "-upload-pack=touch",
        "--depth=1",
        "main; rm -rf /",
        "main'\nimport os",
        "a b",
        "a..b",
        "a//b",
        "a/",
        "a.",
        "topic.lock",
        "main@{1}",
        "a/.hidden",
        "$(id)",
        "x" * 201,
    ],
)
def test_unsafe_refs_are_refused(ref: str) -> None:
    """Options, shell and Python metacharacters and git's forbidden sequences are refused."""
    with pytest.raises(GitRefError):
        validate_git_ref(ref)
    with pytest.raises(ValueError):
        LaunchOptions.from_values(git_ref=ref)


def test_env_manager_ref_defaults_to_the_suite_ref() -> None:
    """One shared ref unless the env-manager ref is given."""
    assert GitRefs.of().suite == "main"
    assert GitRefs.of().env_manager_ref == "main"
    assert GitRefs.of("fix/x").env_manager_ref == "fix/x"
    assert GitRefs.of("fix/x", "main").env_manager_ref == "main"
    with pytest.raises(GitRefError):
        GitRefs.of("main", "-x")


# --- generated notebook source --------------------------------------------------------------


def test_default_bundle_clones_main_and_keeps_the_config_window(tmp_path: Path) -> None:
    """Defaults are unchanged: both clones at main and no session override."""
    bundle = build_kernel_bundle(BundleRequest(_MODEL, "pusher"), tmp_path, _REGISTRY)
    text = _notebook_text(bundle.directory)
    assert _refs_in(text) == ("main", "main")
    assert LIMIT_HOURS_ENV not in text
    assert MARGIN_MINUTES_ENV not in text
    assert "origin/main" not in text
    assert bundle.metadata["is_private"] is True


def test_bundle_carries_refs_and_session_window(tmp_path: Path) -> None:
    """Refs reach the clone cell; the window is exported before the trainer starts."""
    options = LaunchOptions.from_values(
        git_ref="fix/review-remediation", env_manager_ref=_SHA, limit_hours=0.2, margin_minutes=5
    )
    request = BundleRequest(_MODEL, "pusher", launch=options)
    bundle = build_kernel_bundle(request, tmp_path, _REGISTRY)
    text = _notebook_text(bundle.directory)
    assert _refs_in(text) == ("fix/review-remediation", _SHA)
    export = f"os.environ.update({ {LIMIT_HOURS_ENV: '0.2', MARGIN_MINUTES_ENV: '5.0'}!r})\n"
    assert export in text
    assert text.index(export) < text.index("p = subprocess.Popen(cmd")
    assert bundle.metadata["is_private"] is True


@pytest.mark.parametrize("model_key", ["codeformer", "forex_predictor"])
def test_generated_cells_compile(model_key: str) -> None:
    """The clone and training cells with every option are valid Python."""
    meta = resolve_model_metadata(model_key, _REGISTRY)
    options = LaunchOptions.from_values(True, "fix/a", "fix/b", 1.5, 20)
    compile(_cell_text(build_clone_cell("kaggle", options.git_refs)), "clone", "exec")
    compile(_cell_text(build_training_cell(meta, "kaggle", options)), "train", "exec")


@pytest.mark.parametrize(
    ("limit", "margin"), [(0.0, None), (-1.0, None), (None, -1.0), (0.1, 6.0), (1.0, 60.0)]
)
def test_bad_session_windows_are_refused(limit: float | None, margin: float | None) -> None:
    """A non-positive limit, a negative margin or a margin at or past the limit fails."""
    with pytest.raises(ValueError):
        LaunchOptions.from_values(limit_hours=limit, margin_minutes=margin)


# --- the trainer reads the exported window --------------------------------------------------


def test_session_deadline_prefers_the_exported_window() -> None:
    """The exported limit and margin replace config.yaml's kaggle_session."""
    config = {"kaggle_session": {"limit_hours": 12, "margin_minutes": 30}}
    environ = {SESSION_START_ENV: "100", **SessionWindowOverride(0.2, 5).environment()}
    deadline = session_deadline(config, "kaggle", environ, lambda: 100.0)
    assert deadline is not None
    assert deadline.stop_at == pytest.approx(100 + 0.2 * 3600 - 5 * 60)
    only_margin = {SESSION_START_ENV: "0", MARGIN_MINUTES_ENV: "10"}
    deadline = session_deadline(config, "kaggle", only_margin, lambda: 0.0)
    assert deadline is not None
    assert deadline.stop_at == pytest.approx(12 * 3600 - 10 * 60)


@pytest.mark.parametrize(
    "environ", [{LIMIT_HOURS_ENV: "soon"}, {LIMIT_HOURS_ENV: "1", MARGIN_MINUTES_ENV: "90"}]
)
def test_session_deadline_rejects_a_bad_exported_window(environ: dict[str, str]) -> None:
    """A malformed or inconsistent export stops the run instead of being ignored."""
    with pytest.raises(ValueError):
        session_deadline({}, "kaggle", environ, lambda: 0.0)


# --- the generated clone helper against a real git repository -------------------------------


def _run_git(*args: str, cwd: Path) -> str:
    assert _GIT is not None
    return subprocess.run(
        [_GIT, *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / "ecosystem").write_text(name, encoding="utf-8")
    _run_git("add", "ecosystem", cwd=repo)
    _run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name, cwd=repo)
    return _run_git("rev-parse", "HEAD", cwd=repo)


def _sync_repo_helper(folder: Path) -> Any:
    """``_sync_repo`` of the generated clone cell, defined with a token-free git env."""
    text = _cell_text(build_clone_cell("kaggle"))
    start = text.index("def _git(args, cwd=None):")
    helper = folder / "sync_helper.py"
    helper.write_text(text[start : text.index("try:\n", start)], encoding="utf-8")
    scope: dict[str, Any] = {"os": os, "shutil": shutil, "subprocess": subprocess}
    scope["git_env"] = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return runpy.run_path(str(helper), init_globals=scope)["_sync_repo"]


@pytest.mark.skipif(_GIT is None, reason="git is not installed")
def test_clone_helper_checks_out_branch_and_sha(tmp_path: Path) -> None:
    """The helper checks out a branch, then a commit SHA, then main again in place."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _run_git("init", "-q", "-b", "main", cwd=origin)
    _run_git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
    first = _commit(origin, "first")
    _run_git("checkout", "-q", "-b", "fix/branch", cwd=origin)
    _commit(origin, "on-branch")
    _run_git("checkout", "-q", "main", cwd=origin)
    second = _commit(origin, "second")
    sync = _sync_repo_helper(tmp_path)
    url, checkout = origin.as_uri(), tmp_path / "checkout"

    assert sync(url, str(checkout), "ecosystem", "fix/branch") is None
    assert (checkout / "ecosystem").read_text(encoding="utf-8") == "on-branch"
    assert sync(url, str(checkout), "ecosystem", first) is None
    assert _run_git("rev-parse", "HEAD", cwd=checkout) == first
    assert sync(url, str(checkout), "ecosystem", "main") is None
    assert _run_git("rev-parse", "HEAD", cwd=checkout) == second
    error = sync(url, str(checkout), "ecosystem", "no-such-branch")
    assert error is not None and error.startswith("fetch of no-such-branch failed")


# --- every launch path forwards the options -------------------------------------------------


@pytest.mark.usefixtures("kaggle_home")
def test_lemtrain_kaggle_bundle_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``lemtrain kaggle bundle`` writes the refs and the window into the notebook."""
    monkeypatch.setattr(kaggle_commands, "resolve_kaggle_credentials", lambda: ("u", None))
    out = tmp_path / "bundle"
    args = ["bundle", _MODEL, "--out", str(out), "--git-ref", "fix/review-remediation"]
    args += ["--limit-hours", "0.2", "--margin-minutes", "5"]
    result = CliRunner().invoke(kaggle_commands.kaggle_app, args)
    assert result.exit_code == 0, result.output
    text = _notebook_text(out)
    assert _refs_in(text) == ("fix/review-remediation", "fix/review-remediation")
    assert f"'{LIMIT_HOURS_ENV}': '0.2'" in text


@pytest.mark.usefixtures("kaggle_home")
def test_lemtrain_kaggle_bundle_refuses_a_bad_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsafe ref ends the command with an error and writes nothing."""
    monkeypatch.setattr(kaggle_commands, "resolve_kaggle_credentials", lambda: ("u", None))
    out = tmp_path / "bundle"
    args = ["bundle", _MODEL, "--out", str(out), "--git-ref", "main;id"]
    result = CliRunner().invoke(kaggle_commands.kaggle_app, args)
    assert result.exit_code == 1
    assert not out.exists()


def test_cloud_manager_launch_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """``kaggle_cloud_manager --action launch`` passes the refs and the window."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        kaggle_cloud_manager, "launch_kaggle_training", lambda *_a, **kw: calls.append(kw)
    )
    command = ["kaggle_cloud_manager", "--action", "launch", "--model", _MODEL]
    command += ["--git-ref", "fix/a", "--env-manager-ref", "main", "--limit-hours", "2"]
    monkeypatch.setattr(sys, "argv", command)
    kaggle_cloud_manager.main()
    options = calls[0]["options"]
    assert (options.git_refs.suite, options.git_refs.env_manager_ref) == ("fix/a", "main")
    assert options.session == SessionWindowOverride(limit_hours=2.0)


def test_cloud_manager_refuses_a_bad_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad ref is a usage error (exit 2) and nothing is launched."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        kaggle_cloud_manager, "launch_kaggle_training", lambda *_a, **kw: calls.append(kw)
    )
    command = ["kaggle_cloud_manager", "--action", "launch", "--git-ref=-x"]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as exc:
        kaggle_cloud_manager.main()
    assert exc.value.code == 2
    assert not calls


@pytest.mark.usefixtures("kaggle_home")
def test_gui_job_forwards_the_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /kaggle/train`` fields reach the pushed notebook."""
    params = {"accelerator": None, "auto_pull": False, "git_ref": "fix/a", "margin_minutes": 5}
    notebook = run_gui_job(tmp_path, monkeypatch, _MODEL, params)
    assert _refs_in(notebook) == ("fix/a", "fix/a")
    assert f"'{MARGIN_MINUTES_ENV}': '5.0'" in notebook
    assert LIMIT_HOURS_ENV not in notebook


def test_route_accepts_and_validates_the_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fields are optional, recorded in the job params and validated (422)."""
    client, submitted = _client_and_jobs(tmp_path, monkeypatch, "")
    payload = {"model": _MODEL, "git_ref": "fix/a", "limit_hours": 0.2, "margin_minutes": 5}
    with client:
        ok = client.post("/api/training/kaggle/train", json=payload)
        bad_ref = client.post("/api/training/kaggle/train", json={**payload, "git_ref": "a..b"})
        bad_window = client.post(
            "/api/training/kaggle/train", json={**payload, "margin_minutes": 12}
        )
    assert ok.status_code == 200, ok.text
    assert (bad_ref.status_code, bad_window.status_code) == (422, 422)
    assert len(submitted) == 1
    params = submitted[0]["params"]
    assert (params["git_ref"], params["limit_hours"], params["margin_minutes"]) == ("fix/a", 0.2, 5)
    assert params["env_manager_ref"] is None
