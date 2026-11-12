"""One Kaggle push policy for GUI, CLI and notebook (R3-K3 item 2: K-MDL-4, K-MDL-5, K-MDL-6).

kagglehub and the Kaggle metadata API are fakes; nothing reaches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from training.cloud import release_publish
from training.cloud.kaggle_api_sync import KaggleApiSyncService
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.variations import PYTORCH_DEFAULT, VariationKey, selected_variations
from training.cloud.version_notes import build_version_notes, release_facts
from training.export import ExportError
from training.notebooks.cells.sync import build_push_cell
from training.notebooks.registry import resolve_model_metadata
from training.server.app import create_app
from training.version import __version__

_MODEL = "mirnet_exposure"


class _Events:
    """Ordered record of model page syncs and uploads (stands in for the kagglehub module)."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.uploads: list[dict[str, Any]] = []
        self.page_result: dict[str, Any] = {"model": "updated", "variations": {}, "errors": []}

    def model_upload(
        self, handle: str, local_model_dir: str, license_name: str, version_notes: str
    ) -> None:
        """Fake ``kagglehub.model_upload``."""
        self.log.append("upload")
        files = sorted(p.name for p in Path(local_model_dir).rglob("*") if p.is_file())
        self.uploads.append(
            {"handle": handle, "license": license_name, "notes": version_notes, "files": files}
        )

    def upload_count(self) -> int:
        """Number of uploads."""
        return len(self.uploads)


@pytest.fixture(name="events")
def _fixture_events(monkeypatch: pytest.MonkeyPatch) -> _Events:
    events = _Events()
    monkeypatch.setitem(sys.modules, "kagglehub", events)

    def sync_model(_self: KaggleApiSyncService, _metadata: Any) -> dict[str, Any]:
        events.log.append("page")
        return events.page_result

    def sync_metadata(_self: KaggleApiSyncService, _metadata: Any) -> dict[str, Any]:
        events.log.append("metadata")
        return {"model": "updated", "variations": {}, "errors": []}

    monkeypatch.setattr(KaggleApiSyncService, "sync_model", sync_model)
    monkeypatch.setattr(KaggleApiSyncService, "sync_metadata", sync_metadata)
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    return events


@pytest.fixture(name="hub")
def _fixture_hub(fake_workspace: Path) -> Path:
    """A hub with one model folder: an epoch-7 checkpoint, README and metrics."""
    hub = fake_workspace / "LemGendaryModels"
    model_dir = hub / _MODEL
    (model_dir / "checkpoints").mkdir(parents=True)
    torch.save({"epoch": 7, "model_state": {}}, model_dir / "checkpoints" / "latest.pth")
    (model_dir / "README.md").write_text("card", encoding="utf-8")
    (model_dir / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,lr,psnr,ssim\n6,0.5,0.4,0.001,25.0,0.80\n"
        "7,0.4,0.3,0.001,26.5,0.82\n",
        encoding="utf-8",
    )
    return hub


def _manager(hub: Path) -> KaggleModelSyncManager:
    return KaggleModelSyncManager(username="owner", key="KGAT_fake", hub_root=hub)


def test_model_page_before_upload_with_license_and_notes(hub: Path, events: _Events) -> None:
    """The model page is synced first; the upload carries the license and full notes."""
    result = _manager(hub).push_model(_MODEL, variation=PYTORCH_DEFAULT.key)
    assert result.success, result.errors
    assert events.log == ["page", "upload", "metadata"]
    upload = events.uploads[0]
    assert upload["handle"].endswith("/pytorch/default")
    assert upload["license"]
    assert "epoch 7" in upload["notes"]
    assert "psnr=26.5" in upload["notes"]
    assert f"suite v{__version__}" in upload["notes"]
    assert upload["files"] == ["README.md", "latest.pth", "metrics.csv"]


def test_unchanged_payload_is_skipped(hub: Path, events: _Events) -> None:
    """A second push of the same files creates no new version; a changed file does."""
    manager = _manager(hub)
    manager.push_model(_MODEL, variation=PYTORCH_DEFAULT.key)
    again = manager.push_model(_MODEL, variation=PYTORCH_DEFAULT.key)
    assert again.success
    assert again.variations[PYTORCH_DEFAULT.key].skipped
    assert events.upload_count() == 1

    (hub / _MODEL / "metrics.csv").write_text("epoch,val_loss\n8,0.2\n", encoding="utf-8")
    changed = manager.push_model(_MODEL, variation=PYTORCH_DEFAULT.key)
    assert not changed.variations[PYTORCH_DEFAULT.key].skipped
    assert events.upload_count() == 2


def test_failed_model_page_uploads_nothing(hub: Path, events: _Events) -> None:
    """When Kaggle refuses the model page, the push fails before any upload (K-MDL-6)."""
    events.page_result = {"model": None, "variations": {}, "errors": ["403 Forbidden"]}
    result = _manager(hub).push_model(_MODEL, variation=PYTORCH_DEFAULT.key)
    assert not result.success
    assert "403 Forbidden" in result.errors[0]
    assert events.uploads == []


def test_dry_run_touches_no_api(hub: Path, events: _Events) -> None:
    """A dry run neither syncs the model page nor uploads, and stages nothing."""
    result = _manager(hub).push_model(_MODEL, variation=PYTORCH_DEFAULT.key, dry_run=True)
    assert result.variations[PYTORCH_DEFAULT.key].dry_run
    assert "page" not in events.log
    assert events.uploads == []
    assert not (hub / _MODEL / "dist").exists()


def test_version_notes_fall_back_to_metrics_epoch(tmp_path: Path) -> None:
    """Without a readable checkpoint the epoch comes from the last metrics row."""
    (tmp_path / "metrics.csv").write_text("epoch,val_loss,fid\n12,0.1,\n", encoding="utf-8")
    notes = build_version_notes(_MODEL, release_facts(tmp_path), label="ONNX FP16")
    assert notes.startswith(f"Release: {_MODEL} [ONNX FP16] | epoch 12 | val_loss=0.1")
    assert "fid" not in notes


def test_variation_selection_uses_the_enum() -> None:
    """``all`` selects the four keys; legacy names are rejected."""
    assert selected_variations("all") == [k.value for k in VariationKey]
    with pytest.raises(ValueError, match="pytorch/default"):
        selected_variations("pytorch")


def test_routes_accept_only_variation_keys(fake_workspace: Path) -> None:
    """The API validates the variation against the enum (422 for old names) and documents it."""
    client = TestClient(create_app(project_root=fake_workspace, enforce_auth=False))
    bad = client.post(
        "/api/training/kaggle/push-model",
        json={"model": _MODEL, "variation": "onnx_fp16", "dry_run": True},
    )
    assert bad.status_code == 422
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert schema["VariationKey"]["enum"] == [k.value for k in VariationKey]


def test_notebook_release_calls_the_suite() -> None:
    """The publish cell has no upload loop of its own: it calls ``publish_release``."""
    source = "".join(build_push_cell(resolve_model_metadata(_MODEL), "kaggle")["source"])
    assert "publish_release(model_key)" in source
    assert "model_upload" not in source
    assert "except Exception" not in source


def test_release_export_failure_publishes_nothing(
    hub: Path, events: _Events, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed export stops the release before any Kaggle call."""

    def failing_export(model_key: str, output_dir: Path) -> dict[str, Path]:
        raise ExportError("CKPT_MISSING", f"no checkpoint for {model_key} in {output_dir}")

    monkeypatch.setattr(release_publish, "export_all", failing_export)
    result = release_publish.publish_release(_MODEL, manager=_manager(hub))
    assert not result.success
    assert "Export failed" in result.errors[0]
    assert events.log == []
