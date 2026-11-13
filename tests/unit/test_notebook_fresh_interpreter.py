"""Generated notebook cells keep torch and torch-loading suite code out of the kernel process.

The kernel process of a Kaggle or Colab notebook starts with the preinstalled torch. The
dependency cell then installs the suite torch build, so a torch module loaded in that
process is stale, and suite code importing torchvision there fails (``operator
torchvision::nms does not exist``, live run upn-v2-training v2). Every generated training
cell therefore imports neither torch nor suite code that may load it; such code runs in a
fresh interpreter with a checked exit code. The one in-process suite import is the trainer
exit gate, which is checked to be torch-free.
"""

from __future__ import annotations

import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.fixtures.notebook_cells import training_code_sources
from training.cloud import release_publish_cli
from training.cloud.kaggle_model_sync import ModelPushResult, VariationPushResult
from training.notebooks.cells import sync as sync_cells
from training.notebooks.cells.deps import build_install_cell
from training.notebooks.cells.env import build_sentinel_cell
from training.notebooks.cells.sync import build_push_cell
from training.notebooks.registry import load_registry, resolve_model_metadata
from training.platform_paths import PlatformPaths
from training.utils import epoch_probe

_MODELS = ("upn_v2", "nima_aesthetic_mobile", "forex_predictor")
_TORCH_ROOTS = frozenset({"torch", "torchvision", "torchaudio"})
_SUITE_ROOTS = frozenset({"training", "models", "data", "export", "tools"})
_TORCH_FREE_SUITE_MODULES = frozenset({"training.cloud.trainer_exit"})
"""Suite modules a cell may import in the kernel process (checked torch-free below)."""


def _imported_modules(source: str) -> list[str]:
    """Every module an ``import`` statement of ``source`` names (string contents excluded)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
    return found


def _forbidden(module: str) -> bool:
    root = module.split(".")[0]
    if root in _TORCH_ROOTS:
        return True
    return root in _SUITE_ROOTS and module not in _TORCH_FREE_SUITE_MODULES


@pytest.mark.parametrize("model_key", _MODELS)
@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_cells_import_no_torch_in_kernel_process(model_key: str, target: str) -> None:
    """No training notebook cell imports torch or torch-loading suite code in-process."""
    meta = resolve_model_metadata(model_key, load_registry(), {})
    problems = {
        idx: bad
        for idx, source in enumerate(training_code_sources(meta, target))
        if (bad := [m for m in _imported_modules(source) if _forbidden(m)])
    }
    assert not problems


@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_install_cell_probes_torch_in_fresh_interpreter(target: str) -> None:
    """The dependency cell reports the installed torch from a new process."""
    source = "".join(build_install_cell(target)["source"])
    assert not [m for m in _imported_modules(source) if _forbidden(m)]
    assert "version_probe = subprocess.run(" in source


def test_allowed_suite_imports_are_torch_free() -> None:
    """The allowed in-process suite modules load neither torch nor numpy."""
    code = (
        "import importlib, sys\n"
        f"for name in {sorted(_TORCH_FREE_SUITE_MODULES)!r}:\n"
        "    importlib.import_module(name)\n"
        "print(sorted(m for m in ('torch', 'torchvision', 'numpy') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    assert result.stdout.strip() == "[]"


def test_epoch_probe_prints_stored_epochs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The probe maps each path to its stored epoch, ``None`` for unreadable files."""
    good = tmp_path / "latest.pth"
    torch.save({"epoch": 5, "model_state": {"w": torch.zeros(1)}}, good)
    bad = tmp_path / "broken.pth"
    bad.write_bytes(b"not a checkpoint")
    assert epoch_probe.main([str(good), str(bad)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == {str(good): 5, str(bad): None}


def _release(success: bool) -> ModelPushResult:
    variation = VariationPushResult(
        variation="pytorch/default",
        handle="o/m/pytorch/default",
        success=success,
        error=None if success else "upload refused",
    )
    return ModelPushResult(
        model_key="upn_v2",
        success=success,
        variations={"pytorch/default": variation},
        errors=[] if success else ["upload refused"],
    )


@pytest.mark.parametrize("success", [True, False])
def test_release_cli_exit_code(
    success: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The publish CLI prints every variation and exits 1 on a failed release."""
    monkeypatch.setattr(release_publish_cli, "publish_release", lambda key: _release(success))
    code = release_publish_cli.main(["upn_v2"])
    out = capsys.readouterr().out
    if success:
        assert code == release_publish_cli.PUBLISH_OK_EXIT_CODE
        assert "[OK] pytorch/default -> o/m/pytorch/default" in out
        assert "[SUCCESS]" in out
    else:
        assert code == release_publish_cli.PUBLISH_FAILED_EXIT_CODE
        assert "[ERROR] pytorch/default: upload refused" in out


def _stub_suite(suite_dir: Path, publish_code: int) -> None:
    """A checkout whose publish and Drive modules only exit with fixed codes."""
    for package in ("training", "training/cloud"):
        (suite_dir / package).mkdir(parents=True, exist_ok=True)
        (suite_dir / package / "__init__.py").write_text("", encoding="utf-8")
    (suite_dir / "training" / "cloud" / "release_publish_cli.py").write_text(
        f"import sys\nprint('stub publish')\nsys.exit({publish_code})\n", encoding="utf-8"
    )
    (suite_dir / "training" / "gdrive_cloud_manager.py").write_text(
        "import sys\nsys.exit(1)\n", encoding="utf-8"
    )


@pytest.mark.parametrize("publish_code", [0, 1])
def test_push_cell_checks_subprocess_exit_code(
    publish_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed publish subprocess stops the notebook; a failed Drive mirror is reported."""
    paths = PlatformPaths(working_dir=str(tmp_path / "working"), checkout_dir=str(tmp_path))
    monkeypatch.setattr(sync_cells, "platform_paths", lambda target: paths)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    _stub_suite(Path(paths.suite_dir), publish_code)
    meta = resolve_model_metadata("upn_v2", load_registry(), {})
    script = tmp_path / "push_cell.py"
    script.write_text("".join(build_push_cell(meta, "kaggle")["source"]), encoding="utf-8")
    if publish_code:
        with pytest.raises(RuntimeError, match="exit code 1"):
            runpy.run_path(str(script))
    else:
        runpy.run_path(str(script))
        out = capsys.readouterr().out
        assert "stub publish" in out
        assert "Google Drive mirror not updated (exit code 1" in out


@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_sentinel_probe_runs_in_fresh_interpreter(target: str) -> None:
    """The sentinel embeds a compilable torch probe and runs it as ``python -c``."""
    source = "".join(build_sentinel_cell(target)["source"])
    probe = next(
        node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and getattr(node.targets[0], "id", "") == "HW_PROBE"
    )
    assert isinstance(probe, str)
    compile(probe, "hw_probe", "exec")
    assert "torch" in _imported_modules(probe)
    assert "[sys.executable, '-c', HW_PROBE]" in source
