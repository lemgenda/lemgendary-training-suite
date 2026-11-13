"""Kaggle launches forward ``--override-frozen`` to the kernel's trainer when asked (R4-FU 7).

The generated training cell appends the switch to every launch path (single process, DDP,
Forex curriculum); the kernel bundle, the GUI job, ``lemtrain kaggle`` and the
``kaggle_cloud_manager`` CLI pass the request through. Off by default. Nothing is pushed:
the Kaggle API is a fake and credentials are patched (the ``kaggle_home`` fixture).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from training import kaggle_cloud_manager
from training.cli import kaggle_commands
from training.cli_args import OVERRIDE_FROZEN_FLAG
from training.cloud import kernel_launch
from training.cloud.kernel_bundle import BundleRequest, build_kernel_bundle
from training.notebooks.cells.sync import build_training_cell
from training.notebooks.launch_options import LaunchOptions
from training.notebooks.registry import load_registry, resolve_model_metadata
from training.server import job_kaggle_run
from training.server.routes.training import KaggleTrainRequest
from training.server.state import ServerState

_REGISTRY = load_registry()
_SWITCH_LINE = f"cmd += {[OVERRIDE_FROZEN_FLAG]!r}\n"


def _notebook_text(folder: Path) -> str:
    """Source of every code cell of the one notebook in ``folder``."""
    notebook = json.loads(next(folder.glob("*.ipynb")).read_text(encoding="utf-8"))
    return "".join("".join(cell["source"]) for cell in notebook["cells"])


@pytest.mark.parametrize("model_key", ["codeformer", "forex_predictor"])
def test_training_cell_appends_the_switch_only_when_requested(model_key: str) -> None:
    """The switch follows the dispatch, so single, DDP and Forex launches all get it."""
    meta = resolve_model_metadata(model_key, _REGISTRY)
    plain = "".join(build_training_cell(meta, "kaggle")["source"])
    assert OVERRIDE_FROZEN_FLAG not in plain
    source = "".join(build_training_cell(meta, "kaggle", LaunchOptions(True))["source"])
    dispatch_end = source.index("    cmd = [sys.executable, '-u', 'training/train.py'")
    assert source.index(_SWITCH_LINE) > dispatch_end
    assert source.index(_SWITCH_LINE) < source.index("p = subprocess.Popen(cmd")


@pytest.mark.parametrize("override", [False, True])
def test_bundle_notebook_follows_the_request(tmp_path: Path, override: bool) -> None:
    """``BundleRequest.override_frozen`` reaches the bundled notebook."""
    request = BundleRequest("codeformer", "pusher", launch=LaunchOptions(override_frozen=override))
    bundle = build_kernel_bundle(request, tmp_path, _REGISTRY)
    assert (_SWITCH_LINE in _notebook_text(bundle.directory)) is override


class _PushRecorder:
    """Fake ``KaggleApi`` push: keeps the pushed notebook text, then the kernel is cancelled."""

    def __init__(self) -> None:
        self.notebooks: list[str] = []

    def kernels_push(self, folder: str) -> Any:
        """Record the bundled notebook; answer like ``ApiSaveKernelResponse``."""
        self.notebooks.append(_notebook_text(Path(folder)))
        return SimpleNamespace(error="", version_number=1, url="u")

    def kernels_status(self, _handle: str) -> Any:
        """The session is cancelled at once."""
        return SimpleNamespace(status="CANCEL_ACKNOWLEDGED", failure_message="")

    def kernels_logs(self, _handle: str) -> str:
        """No log lines."""
        return "[]"


class _Jobs(job_kaggle_run.KaggleRunJobsMixin):
    """The GUI job mixin with a recording job state."""

    def __init__(self, root: Path) -> None:
        self.hub_root = root
        self.project_root = root
        self.state = cast(ServerState, SimpleNamespace(update_job_status=lambda **_kw: None))
        self._broadcast_log = lambda _job, _line: None


def run_gui_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_key: str, params: dict[str, Any]
) -> str:
    """Run the GUI Kaggle job with ``params`` against a fake API; the pushed notebook text."""
    api = _PushRecorder()
    monkeypatch.setattr(job_kaggle_run, "resolve_kaggle_credentials", lambda: ("u", "KGAT_x"))
    monkeypatch.setattr(job_kaggle_run, "authenticated_kaggle_api", lambda _u, _k: api)
    monkeypatch.setattr(kernel_launch, "kaggle_version_lookup", lambda _api: lambda _h: None)
    event = threading.Event()
    monkeypatch.setattr(event, "wait", lambda _s=None: False)
    _Jobs(tmp_path).execute_kaggle_training("job", model_key, params, event)
    return api.notebooks[0]


@pytest.mark.usefixtures("kaggle_home")
def test_gui_job_forwards_the_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /kaggle/train`` ``override_frozen`` reaches the pushed notebook (default off)."""
    assert KaggleTrainRequest.model_validate({"model": "codeformer"}).override_frozen is False
    params = {"accelerator": None, "auto_pull": False, "override_frozen": True}
    notebook = run_gui_job(tmp_path, monkeypatch, "codeformer", params)
    assert _SWITCH_LINE in notebook


@pytest.mark.usefixtures("kaggle_home")
def test_lemtrain_kaggle_bundle_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``lemtrain kaggle bundle --override-frozen`` writes the switch into the notebook."""
    monkeypatch.setattr(kaggle_commands, "resolve_kaggle_credentials", lambda: ("u", None))
    out = tmp_path / "bundle"
    args = ["bundle", "codeformer", "--out", str(out), "--resume-version", "1"]
    result = CliRunner().invoke(kaggle_commands.kaggle_app, [*args, OVERRIDE_FROZEN_FLAG])
    assert result.exit_code == 0, result.output
    assert _SWITCH_LINE in _notebook_text(out)


@pytest.mark.parametrize(("argv", "expected"), [([], False), ([OVERRIDE_FROZEN_FLAG], True)])
def test_cloud_manager_launch_option(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: bool
) -> None:
    """``kaggle_cloud_manager --action launch`` passes the switch to the launch."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        kaggle_cloud_manager, "launch_kaggle_training", lambda *_a, **kw: calls.append(kw)
    )
    command = ["kaggle_cloud_manager", "--action", "launch", "--model", "codeformer", *argv]
    monkeypatch.setattr(sys, "argv", command)
    kaggle_cloud_manager.main()
    assert calls[0]["options"].override_frozen is expected
