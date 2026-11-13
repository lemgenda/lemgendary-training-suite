"""The training cell stops the notebook when the trainer exits non-zero.

The generated training cell is executed against a temporary suite whose ``training/train.py``
stub exits with a chosen code; nothing reaches the network or a GPU. A non-zero exit raises
``TrainerExitError`` (so the publish cell never runs); exit 0 with the session deadline
marker is the intentional deadline stop and is not a failure.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from training.cloud.session_deadline import DEADLINE_MARKER
from training.cloud.trainer_exit import (
    TRAINER_OK_EXIT_CODE,
    TrainerExitError,
    TrainerOutcome,
    check_trainer_exit,
)
from training.notebooks.builders.kaggle import build_training_notebook
from training.notebooks.cells import sync as sync_cells
from training.notebooks.registry import load_registry, resolve_model_metadata
from training.platform_paths import PlatformPaths
from training.run_settings.refusal import RESUME_REFUSAL_RECORD

_KEY = "nima_aesthetic_mobile"
_REFUSAL_TEXT = "Frozen settings changed: batch_size"


def _source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def _write_refusal(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {"error": _REFUSAL_TEXT, "settings": ["batch_size"]}
    (model_dir / RESUME_REFUSAL_RECORD).write_text(json.dumps(payload), encoding="utf-8")


def _write_marker(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / DEADLINE_MARKER).write_text('{"epoch": 3}', encoding="utf-8")


class TestCheckTrainerExit:
    """The exit classification the notebook imports."""

    def test_zero_without_marker_is_completed(self, tmp_path: Path) -> None:
        """Exit 0 without the marker is a completed run."""
        assert check_trainer_exit(TRAINER_OK_EXIT_CODE, tmp_path) is TrainerOutcome.COMPLETED

    def test_zero_with_marker_is_deadline_stop(self, tmp_path: Path) -> None:
        """Exit 0 with the marker is the deadline stop, not a failure."""
        _write_marker(tmp_path)
        assert check_trainer_exit(0, tmp_path) is TrainerOutcome.DEADLINE_STOP

    def test_non_zero_raises_without_refusal(self, tmp_path: Path) -> None:
        """A non-zero exit without a refusal record names the exit code."""
        with pytest.raises(TrainerExitError, match="code 1: see the training log") as caught:
            check_trainer_exit(1, tmp_path)
        assert caught.value.returncode == 1
        assert caught.value.refusal is None

    def test_non_zero_carries_refusal_and_keeps_record(self, tmp_path: Path) -> None:
        """A refused resume raises with the record text; the record stays for the watcher."""
        _write_refusal(tmp_path)
        with pytest.raises(TrainerExitError, match=_REFUSAL_TEXT) as caught:
            check_trainer_exit(1, tmp_path)
        assert caught.value.refusal == _REFUSAL_TEXT
        assert (tmp_path / RESUME_REFUSAL_RECORD).is_file()

    def test_non_zero_with_marker_is_still_a_failure(self, tmp_path: Path) -> None:
        """The marker never excuses a non-zero exit."""
        _write_marker(tmp_path)
        with pytest.raises(TrainerExitError):
            check_trainer_exit(-9, tmp_path)


def test_gate_sits_in_training_cell_before_publish() -> None:
    """The Kaggle notebook gates the trainer exit in the training cell, before publish."""
    meta = resolve_model_metadata(_KEY, load_registry(), {})
    sources = [_source(c) for c in build_training_notebook(meta)["cells"]]
    training = next(i for i, s in enumerate(sources) if "subprocess.Popen(cmd" in s)
    publish = next(i for i, s in enumerate(sources) if "publish_release(model_key)" in s)
    assert training < publish
    gate = sources[training]
    assert "check_trainer_exit(p.wait(), Path(" in gate
    assert f"/LemGendaryModels/{_KEY}" in gate
    assert "except Exception" not in gate
    assert "    pass\n" not in gate


@pytest.fixture(name="suite")
def _suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlatformPaths:
    """Temporary platform folders with a suite checkout whose trainer is a stub."""
    paths = PlatformPaths(working_dir=str(tmp_path / "working"), checkout_dir=str(tmp_path))
    monkeypatch.setattr(sync_cells, "platform_paths", lambda target: paths)
    (Path(paths.suite_dir) / "training").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return paths


def _run_cell(paths: PlatformPaths, target: str, exit_code: int) -> None:
    """Run the generated training cell with a trainer stub exiting ``exit_code``."""
    stub = Path(paths.suite_dir) / "training" / "train.py"
    stub.write_text(f"import sys\nsys.exit({exit_code})\n", encoding="utf-8")
    meta = resolve_model_metadata(_KEY, load_registry(), {})
    script = Path(paths.checkout_dir) / f"training_cell_{target}.py"
    script.write_text(_source(sync_cells.build_training_cell(meta, target)), encoding="utf-8")
    runpy.run_path(str(script))


def _model_dir(paths: PlatformPaths) -> Path:
    return Path(paths.hub_root) / _KEY


@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_failed_training_stops_notebook(suite: PlatformPaths, target: str) -> None:
    """A failed trainer raises out of the cell on both platforms."""
    with pytest.raises(TrainerExitError, match="code 3"):
        _run_cell(suite, target, 3)


def test_refused_resume_stops_notebook_with_refusal_text(suite: PlatformPaths) -> None:
    """A refused resume raises with the refusal text and keeps the record."""
    _write_refusal(_model_dir(suite))
    with pytest.raises(TrainerExitError, match=_REFUSAL_TEXT):
        _run_cell(suite, "kaggle", 1)
    assert (_model_dir(suite) / RESUME_REFUSAL_RECORD).is_file()


def test_deadline_stop_does_not_fail(
    suite: PlatformPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 with the deadline marker passes and reports the deadline stop."""
    _write_marker(_model_dir(suite))
    _run_cell(suite, "kaggle", 0)
    assert "[DEADLINE] Training stopped at the session deadline" in capsys.readouterr().out


def test_completed_training_passes(
    suite: PlatformPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 0 without the marker passes."""
    _run_cell(suite, "colab", 0)
    assert "[OK] Training finished with exit code 0." in capsys.readouterr().out
