"""Unit and failure-injection tests for Kaggle model push and pull resilience.

Verifies:
1. Idempotent push (no-op when hashes match).
2. Force push (--force bypasses no-op check).
3. Push retry with exponential backoff on transient errors.
4. Push failure handling when retries are exhausted.
5. Pull version pinning (@v2, --version 2).
6. Pull checksum verification and error on corrupted download.
7. Machine-readable sync report generation (--report, --json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from training.cli.lemtrain import app
from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_api_sync import KaggleApiSyncService
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.variations import PYTORCH_DEFAULT

runner = CliRunner()
_MODEL = "mirnet_exposure"


def _save_ckpt(path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": {"w": torch.zeros(1)}}, path)


@pytest.fixture(name="hub_env")
def _fixture_hub_env(fake_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare local model dir in LemGendaryModels."""
    model_dir = fake_workspace / "LemGendaryModels" / _MODEL
    _save_ckpt(model_dir / "checkpoints" / "latest.pth", 5)
    _save_ckpt(model_dir / "checkpoints" / "best.pth", 5)
    (model_dir / "README.md").write_text("# Model Readme\n", encoding="utf-8")
    (model_dir / "metrics.csv").write_text("epoch,loss\n5,0.1\n", encoding="utf-8")

    # Mock Kaggle metadata API
    def sync_model(_self: Any, _metadata: Any) -> dict[str, Any]:
        return {"model": "updated", "variations": {}, "errors": []}

    def sync_metadata(_self: Any, _metadata: Any) -> dict[str, Any]:
        return {"model": "updated", "variations": {}, "errors": []}

    monkeypatch.setattr(KaggleApiSyncService, "sync_model", sync_model)
    monkeypatch.setattr(KaggleApiSyncService, "sync_metadata", sync_metadata)

    return fake_workspace / "LemGendaryModels"


def test_push_idempotent_and_force(hub_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify push skips unchanged payloads on second run unless force=True."""
    upload_calls: list[str] = []

    class FakeKaggleHub:
        """Fake KaggleHub client recording uploads."""

        def model_upload(
            self, handle: str, local_model_dir: str, license_name: str, version_notes: str
        ) -> None:
            """Record upload handle."""
            del local_model_dir, license_name, version_notes
            upload_calls.append(handle)

        def ping(self) -> bool:
            """Stub health check."""
            return True

    monkeypatch.setitem(sys.modules, "kagglehub", FakeKaggleHub())

    mgr = KaggleModelSyncManager(hub_root=hub_env)

    # First push: should upload
    res1 = mgr.push_variation(_MODEL, PYTORCH_DEFAULT.key)
    assert res1.success
    assert not res1.skipped
    assert len(upload_calls) == 1

    # Second push: should skip (idempotent no-op)
    res2 = mgr.push_variation(_MODEL, PYTORCH_DEFAULT.key)
    assert res2.success
    assert res2.skipped
    assert len(upload_calls) == 1

    # Third push with force=True: should re-upload
    res3 = mgr.push_variation(_MODEL, PYTORCH_DEFAULT.key, force=True)
    assert res3.success
    assert not res3.skipped
    assert len(upload_calls) == 2


def test_push_retry_on_transient_error(hub_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify push retries on transient network/Kaggle errors with exponential backoff."""
    attempts = 0

    class FailingThenSucceedingHub:
        """Fake KaggleHub failing twice before succeeding."""

        def model_upload(
            self, handle: str, local_model_dir: str, license_name: str, version_notes: str
        ) -> None:
            """Simulate transient upload failure."""
            del handle, local_model_dir, license_name, version_notes
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                # Raise an error matching kagglehub_errors
                err_cls = kagglehub_errors()[0]
                raise err_cls(f"Simulated network glitch attempt {attempts}")

        def ping(self) -> bool:
            """Stub health check."""
            return True

    monkeypatch.setitem(sys.modules, "kagglehub", FailingThenSucceedingHub())
    monkeypatch.setattr("time.sleep", lambda _s: None)  # fast tests

    mgr = KaggleModelSyncManager(hub_root=hub_env)
    res = mgr.push_variation(_MODEL, PYTORCH_DEFAULT.key)

    assert res.success
    assert res.retries == 2
    assert attempts == 3


def test_push_exhausted_retries_failure(hub_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify push fails gracefully when retries are exhausted."""
    attempts = 0

    class AlwaysFailingHub:
        """Fake KaggleHub that always raises."""

        def model_upload(
            self, handle: str, local_model_dir: str, license_name: str, version_notes: str
        ) -> None:
            """Simulate persistent upload failure."""
            del handle, local_model_dir, license_name, version_notes
            nonlocal attempts
            attempts += 1
            err_cls = kagglehub_errors()[0]
            raise err_cls("Persistent network outage")

        def ping(self) -> bool:
            """Stub health check."""
            return False

    monkeypatch.setitem(sys.modules, "kagglehub", AlwaysFailingHub())
    monkeypatch.setattr("time.sleep", lambda _s: None)

    mgr = KaggleModelSyncManager(hub_root=hub_env)
    res = mgr.push_variation(_MODEL, PYTORCH_DEFAULT.key)

    assert not res.success
    assert "Persistent network outage" in (res.error or "")
    assert res.retries == 3
    assert attempts == 4


def test_pull_checksum_verification_and_corruption_detection(
    hub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pull validates checksums and raises ValueError on corrupted downloaded files."""
    class FakeDownloadHub:
        """Fake KaggleHub for download checksum tests."""

        def model_download(self, handle: str, output_dir: str) -> str:
            """Write sample downloaded artifacts."""
            del handle
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoints").mkdir(parents=True, exist_ok=True)
            _save_ckpt(out / "checkpoints" / "latest.pth", 10)
            (out / "README.md").write_text("Downloaded Readme\n", encoding="utf-8")
            (out / "metrics.csv").write_text("epoch,loss\n10,0.05\n", encoding="utf-8")
            return output_dir

        def ping(self) -> bool:
            """Stub health check."""
            return True

    monkeypatch.setitem(sys.modules, "kagglehub", FakeDownloadHub())

    mgr = KaggleModelSyncManager(hub_root=hub_env)

    # Normal pull: verifies and succeeds
    pull_res = mgr.pull_variation_detailed(_MODEL, PYTORCH_DEFAULT.key, force_download=True)
    assert pull_res.success
    assert pull_res.checksum_verified
    assert len(pull_res.checksums) >= 3

    # Pull with expected checksum mismatch: should detect corruption and fail
    bad_expected = {"README.md": "0000000000000000000000000000000000000000000000000000000000000000"}
    corrupt_res = mgr.pull_variation_detailed(
        _MODEL,
        PYTORCH_DEFAULT.key,
        force_download=True,
        expected_checksums=bad_expected,
    )
    assert not corrupt_res.success
    assert "Integrity check failed" in (corrupt_res.error or "")


def test_pull_version_pinning(hub_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pull appends pinned version to Kaggle handle."""
    handles: list[str] = []

    class FakeDownloadHub:
        """Fake KaggleHub recording requested handles."""

        def model_download(self, handle: str, output_dir: str) -> str:
            """Record handle and write README."""
            handles.append(handle)
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "README.md").write_text("Readme v2\n", encoding="utf-8")
            return output_dir

        def ping(self) -> bool:
            """Stub health check."""
            return True

    monkeypatch.setitem(sys.modules, "kagglehub", FakeDownloadHub())

    mgr = KaggleModelSyncManager(hub_root=hub_env)
    res = mgr.pull_variation_detailed(
        _MODEL, PYTORCH_DEFAULT.key, force_download=True, version=2
    )

    assert res.success
    assert len(handles) == 1
    assert handles[0].endswith("/2")


def test_cli_sync_push_and_pull_reports(
    hub_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI sync push and sync pull emit reports via --report and --json."""
    class FakeKaggleHub:
        """Fake KaggleHub for CLI sync push/pull tests."""

        def model_upload(
            self, handle: str, local_model_dir: str, license_name: str, version_notes: str
        ) -> None:
            """No-op upload."""
            del handle, local_model_dir, license_name, version_notes

        def model_download(self, handle: str, output_dir: str) -> str:
            """Write sample CLI pulled artifacts."""
            del handle
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "checkpoints").mkdir(parents=True, exist_ok=True)
            _save_ckpt(out / "checkpoints" / "latest.pth", 12)
            (out / "README.md").write_text("CLI Pulled Readme\n", encoding="utf-8")
            (out / "metrics.csv").write_text("epoch,loss\n12,0.01\n", encoding="utf-8")
            return output_dir

    monkeypatch.setitem(sys.modules, "kagglehub", FakeKaggleHub())

    push_report_file = tmp_path / "push_report.json"
    pull_report_file = tmp_path / "pull_report.json"

    # Run CLI push with --report and --json
    push_args = [
        "sync",
        "push",
        _MODEL,
        "--variation",
        "pytorch/default",
        "--source-dir",
        str(hub_env),
        "--report",
        str(push_report_file),
        "--json",
    ]
    push_res = runner.invoke(app, push_args)
    assert push_res.exit_code == 0
    assert push_report_file.is_file()

    push_data = json.loads(push_report_file.read_text(encoding="utf-8"))
    assert push_data["action"] == "push"
    assert push_data["model_key"] == _MODEL
    assert push_data["success"] is True
    assert "pytorch/default" in push_data["variations"]

    # Run CLI pull with --version, --report, and --json
    pull_args = [
        "sync",
        "pull",
        _MODEL,
        "--variation",
        "pytorch/default",
        "--version",
        "2",
        "--force",
        "--report",
        str(pull_report_file),
        "--json",
    ]
    # Point hub_root in KaggleModelSyncManager
    monkeypatch.setattr(
        KaggleModelSyncManager,
        "__init__",
        lambda self, username=None, key=None, hub_root=None: setattr(
            self, "hub_root", hub_env
        )
        or setattr(self, "username", "lemgenda")
        or setattr(self, "registry", MagicMock(get_model=lambda _k: None))
        or setattr(
            self, "validator", MagicMock(validate_all=lambda *a, **k: MagicMock(is_valid=True))
        )
        or setattr(self, "metadata_service", MagicMock()),
    )

    pull_res = runner.invoke(app, pull_args)
    assert pull_res.exit_code == 0
    assert pull_report_file.is_file()

    pull_data = json.loads(pull_report_file.read_text(encoding="utf-8"))
    assert pull_data["action"] == "pull"
    assert pull_data["model_key"] == _MODEL
    assert pull_data["success"] is True
    assert pull_data["variations"]["pytorch/default"]["version"] == 2
    assert pull_data["variations"]["pytorch/default"]["checksum_verified"] is True
