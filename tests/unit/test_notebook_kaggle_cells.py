"""Generated Kaggle notebook cells: token-free clone, epoch-based recovery, forex output.

The recovery cell is executed against a temporary hub with a fake kagglehub; nothing reaches
the network. Paths come from ``training.platform_paths`` (patched to temp folders).
"""

from __future__ import annotations

import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from training.ecosystem import github_repo_url
from training.notebooks.builders.kaggle import generate_training_notebook
from training.notebooks.cells import model as model_cells
from training.notebooks.cells.inference import build_pth_cell
from training.notebooks.cells.repo import build_clone_cell
from training.notebooks.registry import load_registry, resolve_model_metadata
from training.platform_paths import PlatformPaths, platform_paths
from training.utils.model_payload import checkpoint_epoch

_KEY = "nima_aesthetic_mobile"
_SUITE_ROOT = Path(__file__).resolve().parents[2]


def _source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def _clone_targets(text: str) -> list[str]:
    return re.findall(r"^(?:suite_path|env_mgr_path|checkout_dir) = '([^']*)'", text, re.MULTILINE)


class TestCloneCellSecurity:
    """K-NB-4: the PAT never lands in a URL or a file, and nothing is cloned to the output."""

    @pytest.mark.parametrize("target", ["kaggle", "colab"])
    def test_no_token_in_urls(self, target: str) -> None:
        """Clone URLs are the plain ecosystem URLs; no credential form appears anywhere."""
        text = _source(build_clone_cell(target))
        assert f"suite_url = {github_repo_url('lemgendary-training-suite')!r}" in text
        for forbidden in ("x-access-token:", "{pat", "{_safe_pat", "{token}@", "_url_quote"):
            assert forbidden not in text
        assert not re.search(r"https://[^'\s]*@", text)
        assert "GIT_ASKPASS" in text
        assert "['remote', 'set-url', 'origin', url]" in text

    def test_kaggle_clone_outside_working(self) -> None:
        """Kaggle checkouts live in the session-only folder, never under /kaggle/working."""
        targets = _clone_targets(_source(build_clone_cell("kaggle")))
        assert len(targets) == 3
        assert all(t.startswith(platform_paths("kaggle").checkout_dir) for t in targets)
        assert not any(t.startswith("/kaggle/working") for t in targets)

    def test_generated_notebook(self, tmp_path: Path) -> None:
        """The whole generated Kaggle training notebook carries no token or working clone."""
        registry = load_registry()
        path = generate_training_notebook(_KEY, str(tmp_path), registry, {})
        assert path is not None
        text = "".join(
            "".join(c["source"])
            for c in json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
        )
        assert "x-access-token:" not in text
        assert not re.search(r"https://[^'\s]*@", text)
        assert "/kaggle/working/lemgendary-" not in text
        assert all(t.startswith("/kaggle/temp") for t in _clone_targets(text))


def _save(path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": {"w": torch.zeros(1)}}, path)


def _fake_kagglehub(download_dir: Path) -> Any:
    names = (
        "BackendError",
        "NotFoundError",
        "CredentialError",
        "UnauthenticatedError",
        "DataCorruptionError",
        "KaggleEnvironmentError",
    )
    exceptions = SimpleNamespace(**{n: type(n, (Exception,), {}) for n in names})
    return SimpleNamespace(exceptions=exceptions, model_download=lambda handle: str(download_dir))


@pytest.fixture(name="recovery")
def _recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Hub latest download (no model key in the file names) and an older local hub."""
    paths = PlatformPaths(working_dir=str(tmp_path / "working"), checkout_dir=str(tmp_path))
    monkeypatch.setattr(model_cells, "platform_paths", lambda target: paths)
    download = tmp_path / "hub-download"
    _save(download / "checkpoints" / "latest.pth", 7)
    _save(download / "checkpoints" / "best.pth", 3)
    (download / "metrics.csv").write_text("epoch\n7\n", encoding="utf-8")
    ckpt_dir = Path(paths.hub_root) / _KEY / "checkpoints"
    _save(ckpt_dir / "best.pth", 4)
    _save(ckpt_dir / "latest.pth", 2)
    fake = _fake_kagglehub(download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    monkeypatch.setitem(sys.modules, "kagglehub.exceptions", fake.exceptions)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # The cell runs the epoch probe from the (empty) temporary checkout; the real suite
    # package resolves through PYTHONPATH in that fresh interpreter.
    Path(paths.suite_dir).mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", str(_SUITE_ROOT))
    return {"ckpt_dir": ckpt_dir, "model_dir": ckpt_dir.parent}


def test_recovery_uses_epochs_and_slug(recovery: dict[str, Path]) -> None:
    """Hub files are adopted without a key-in-name filter, by stored epoch only."""
    meta = resolve_model_metadata(_KEY, load_registry(), {})
    cell = model_cells.build_checkpoint_recovery_cell(meta, "kaggle")
    script = recovery["model_dir"].parent.parent / "recovery_cell.py"
    script.write_text(_source(cell), encoding="utf-8")
    runpy.run_path(str(script))

    ckpt_dir = recovery["ckpt_dir"]
    assert checkpoint_epoch(ckpt_dir / "latest.pth") == 7
    assert checkpoint_epoch(ckpt_dir / "best.pth") == 4
    assert (recovery["model_dir"] / "metrics.csv").read_text(encoding="utf-8") == "epoch\n7\n"


def test_stealth_cell_considers_pt_files() -> None:
    """The local loader globs ``.pth`` and ``.pt`` and picks by stored epoch."""
    meta = resolve_model_metadata(_KEY, load_registry(), {})
    text = _source(model_cells.build_stealth_cell(meta, "kaggle"))
    assert "'*.pth', '*.pt'" in text
    assert "checkpoint_epoch" in text
    assert "getsize" not in text


def test_forex_pt_cell_indexes_output_dict() -> None:
    """ForexPredictor returns a dict; the usage cell indexes it like the model card."""
    meta = resolve_model_metadata("forex_predictor", load_registry(), {})
    text = _source(build_pth_cell(meta))
    assert "direction_logits = out['direction_logits']" in text
    assert "direction_logits, tp_sl_pips = model(" not in text
