"""Kaggle pulls merge into a non-empty hub folder and never delete local work.

kagglehub is replaced by a fake that behaves like the real one on ``output_dir``: it refuses
a non-empty folder and, with ``force_download``, empties it first. No network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.model_pull import merge_downloaded_payload
from training.cloud.variations import PYTORCH_DEFAULT, PYTORCH_FP32
from training.utils.model_payload import checkpoint_epoch

_MODEL = "nima_aesthetic_mobile"


def _save_ckpt(path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": {"w": torch.zeros(1)}}, path)


class _FakeKagglehub:
    """Writes ``files`` (relative path -> writer) into ``output_dir`` like kagglehub does."""

    def __init__(self, files: dict[str, Any]) -> None:
        self.files = files
        self.handles: list[str] = []

    def model_download(self, handle: str, output_dir: str, force_download: bool = False) -> str:
        """Refuse a non-empty folder unless forced, then write the payload."""
        self.handles.append(handle)
        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            if not force_download:
                raise FileExistsError(f"output_dir is not empty: {output_dir}")
            for child in out.iterdir():
                child.unlink()
        out.mkdir(parents=True, exist_ok=True)
        for rel, writer in self.files.items():
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            writer(target)
        return output_dir

    def handle_count(self) -> int:
        """Number of downloads requested."""
        return len(self.handles)


def _text(content: str) -> Any:
    return lambda path: path.write_text(content, encoding="utf-8")


def _ckpt(epoch: int) -> Any:
    return lambda path: _save_ckpt(path, epoch)


@pytest.fixture(name="hub")
def _hub(fake_workspace: Path) -> Path:
    """A hub root holding local work for the model."""
    model_dir = fake_workspace / "LemGendaryModels" / _MODEL
    _save_ckpt(model_dir / "checkpoints" / "latest.pth", 5)
    _save_ckpt(model_dir / "checkpoints" / "best.pth", 4)
    (model_dir / "notebooks").mkdir(parents=True)
    (model_dir / "notebooks" / "local.ipynb").write_text("{}", encoding="utf-8")
    (model_dir / "metrics.csv").write_text("epoch\n5\n", encoding="utf-8")
    return fake_workspace / "LemGendaryModels"


def _manager(hub: Path, monkeypatch: pytest.MonkeyPatch, fake: _FakeKagglehub) -> Any:
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        # setenv records the original state, so the exported credentials are undone later.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    return KaggleModelSyncManager(username="owner", key="KGAT_fake", hub_root=hub)


def _remote_default() -> dict[str, Any]:
    return {
        "checkpoints/latest.pth": _ckpt(7),
        "checkpoints/best.pth": _ckpt(3),
        "metrics.csv": _text("epoch\n7\n"),
        "notebooks/remote.ipynb": _text("{}"),
        "dist/kaggle/stray.onnx": _text("x"),
        "checkpoints/stray.pt": _text("x"),
    }


def test_pull_into_non_empty_folder_merges(hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Newer remote checkpoints win by epoch; older ones and local files stay."""
    fake = _FakeKagglehub(_remote_default())
    mgr = _manager(hub, monkeypatch, fake)
    assert mgr.pull_variation(_MODEL, PYTORCH_DEFAULT.key)

    model_dir = hub / _MODEL
    assert checkpoint_epoch(model_dir / "checkpoints" / "latest.pth") == 7
    assert checkpoint_epoch(model_dir / "checkpoints" / "best.pth") == 4
    assert (model_dir / "metrics.csv").read_text(encoding="utf-8") == "epoch\n7\n"
    assert (model_dir / "notebooks" / "local.ipynb").is_file()
    assert not (model_dir / "notebooks" / "remote.ipynb").exists()
    assert not (model_dir / "dist").exists()
    assert not (model_dir / "checkpoints" / "stray.pt").exists()
    assert fake.handles == [f"owner/{mgr.get_model_slug(_MODEL)}/pytorch/default"]
    assert not [p for p in hub.iterdir() if p.name.startswith(".kaggle-pull-")]


def test_force_overwrites_only_pulled_files(hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``force`` replaces the pulled files and leaves every other local file in place."""
    mgr = _manager(hub, monkeypatch, _FakeKagglehub(_remote_default()))
    assert mgr.pull_variation(_MODEL, PYTORCH_DEFAULT.key, force_download=True)

    model_dir = hub / _MODEL
    assert checkpoint_epoch(model_dir / "checkpoints" / "best.pth") == 3
    assert (model_dir / "notebooks" / "local.ipynb").is_file()
    assert sorted(p.name for p in (model_dir / "checkpoints").iterdir()) == [
        "best.pth",
        "latest.pth",
    ]


def test_repeated_auto_pull_works(hub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second pull (the per-epoch auto-pull) into the now non-empty folder succeeds."""
    fake = _FakeKagglehub(_remote_default())
    mgr = _manager(hub, monkeypatch, fake)
    assert mgr.pull_variation(_MODEL, PYTORCH_DEFAULT.key)
    assert mgr.pull_variation(_MODEL, PYTORCH_DEFAULT.key)
    assert fake.handle_count() == 2


def test_export_variation_keeps_local_file_unless_forced(
    hub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Payloads without checkpoints keep an existing local file unless forced."""
    mgr = _manager(hub, monkeypatch, _FakeKagglehub({"Model.pt": _text("remote")}))
    dest = mgr.pull_destination(_MODEL, PYTORCH_FP32.key)
    dest.mkdir(parents=True)
    (dest / "Model.pt").write_text("local", encoding="utf-8")
    (dest / "keep.txt").write_text("mine", encoding="utf-8")

    assert mgr.pull_variation(_MODEL, PYTORCH_FP32.key)
    assert (dest / "Model.pt").read_text(encoding="utf-8") == "local"
    assert mgr.pull_variation(_MODEL, PYTORCH_FP32.key, force_download=True)
    assert (dest / "Model.pt").read_text(encoding="utf-8") == "remote"
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_merge_uses_epoch_not_size(tmp_path: Path) -> None:
    """A bigger but older checkpoint never replaces a newer local one."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "checkpoints").mkdir(parents=True)
    (dst / "checkpoints").mkdir(parents=True)
    (src / "checkpoints" / "latest.pth").write_bytes(b"x" * 100)
    (dst / "checkpoints" / "latest.pth").write_bytes(b"y")
    epochs = {src / "checkpoints" / "latest.pth": 2, dst / "checkpoints" / "latest.pth": 4}

    report = merge_downloaded_payload(src, dst, force=False, epoch_of=epochs.get)
    assert report.kept == ["checkpoints/latest.pth"]
    assert (dst / "checkpoints" / "latest.pth").read_bytes() == b"y"
