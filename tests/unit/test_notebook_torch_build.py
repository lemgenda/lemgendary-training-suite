"""The dependency cell installs the cloud CUDA torch build from its own index.

Live run upn-v2-training v2 named the ``cu126`` index but got the CUDA 13 torch: the manifest
pins ``torch==2.15.0``, which has no ``cu126`` wheel, so pip took it from PyPI. The cell is
executed here with ``nvidia-smi`` present and ``pip`` recorded instead of run.
"""

from __future__ import annotations

import runpy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from training.notebooks.cells import deps as deps_cells
from training.notebooks.cells.env import build_sentinel_cell
from training.notebooks.torch_build import CLOUD_CUDA_BUILD, TORCH_FAMILY
from training.platform_paths import PlatformPaths

_MANIFEST = (
    "torch==2.15.0\n"
    "torchvision==0.29.2\n"
    "torchaudio==2.11.1\n"
    "torchmetrics==1.9.0\n"
    "torch-fidelity>=0.3.0\n"
    "timm==1.0.30  # vision backbones\n"
)


class _PipRecorder:
    """Stands in for ``subprocess.run``: records pip calls, answers the torch probe."""

    def __init__(self) -> None:
        self.pip_calls: list[list[str]] = []
        self.files: dict[str, str] = {}

    def __call__(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if "pip" in cmd:
            self.pip_calls.append(list(cmd))
            for flag in ("-r", "-c"):
                if flag in cmd:
                    path = cmd[cmd.index(flag) + 1]
                    self.files[flag] = Path(path).read_text(encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "2.14.1+cu126 on CUDA 12.6\n", "")

    def requirement_lines(self, flag: str) -> list[str]:
        """Lines of the file the manifest install passed with ``flag`` (``-r`` or ``-c``)."""
        return self.files[flag].splitlines()


@pytest.fixture(name="recorder")
def _recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PipRecorder:
    """Run the Kaggle dependency cell against a temporary manifest on a CUDA host."""
    paths = PlatformPaths(working_dir=str(tmp_path / "working"), checkout_dir=str(tmp_path))
    monkeypatch.setattr(deps_cells, "platform_paths", lambda target: paths)
    manifest = Path(paths.env_mgr_dir) / "requirements" / "requirements-training.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(_MANIFEST, encoding="utf-8")
    recorder = _PipRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    monkeypatch.setattr(shutil, "which", lambda name: name if name == "nvidia-smi" else None)
    script = tmp_path / "deps_cell.py"
    script.write_text("".join(deps_cells.build_install_cell("kaggle")["source"]), encoding="utf-8")
    runpy.run_path(str(script))
    return recorder


def test_torch_family_comes_from_the_build_index_only(recorder: _PipRecorder) -> None:
    """The first pip call installs the exact pins with ``--index-url`` (no PyPI)."""
    first = recorder.pip_calls[0]
    assert first[first.index("--index-url") + 1] == CLOUD_CUDA_BUILD.index_url
    assert first[-len(CLOUD_CUDA_BUILD.pins) :] == list(CLOUD_CUDA_BUILD.pins)
    assert "--extra-index-url" not in first


def test_manifest_install_drops_and_constrains_the_torch_family(recorder: _PipRecorder) -> None:
    """The manifest is installed without its torch pins, constrained to the build."""
    assert len(recorder.pip_calls) == 2
    kept = recorder.requirement_lines("-r")
    names = {line.split("=")[0].split(">")[0].strip() for line in kept}
    assert not names & set(TORCH_FAMILY)
    assert {"torchmetrics", "torch-fidelity", "timm"} <= names
    assert recorder.requirement_lines("-c") == list(CLOUD_CUDA_BUILD.pins)


def test_sentinel_names_no_other_torch_build() -> None:
    """No cell hardcodes a torch build besides :data:`CLOUD_CUDA_BUILD`."""
    for target in ("kaggle", "colab"):
        source = "".join(build_sentinel_cell(target)["source"])
        assert "torch==" not in source
