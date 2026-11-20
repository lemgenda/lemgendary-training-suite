"""Unit and regression tests for environment preflight and cloud pipeline resilience."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.cloud.session_deadline import (
    SESSION_START_ENV,
    session_deadline,
)
from training.environment.preflight import (
    EnvironmentPreflight,
)
from training.hardware.discovery import DeviceInfo
from training.platform_paths import (
    on_colab,
    on_kaggle,
    platform_paths,
)


def test_platform_paths_resolution() -> None:
    """Platform paths resolve correctly for kaggle and colab."""
    k_paths = platform_paths("kaggle")
    assert k_paths.working_dir == "/kaggle/working"
    assert k_paths.checkout_dir == "/kaggle/temp"
    assert k_paths.input_dir == "/kaggle/input"

    c_paths = platform_paths("colab")
    assert c_paths.working_dir == "/content"
    assert c_paths.checkout_dir == "/content"
    assert c_paths.input_dir == "/content/drive/MyDrive"


def test_on_colab_and_kaggle_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime flags detect colab vs kaggle vs local correctly."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_WORKING_DIR", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    assert not on_kaggle()
    assert not on_colab()

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert on_kaggle()
    assert not on_colab()

    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-2026-11")
    assert on_colab()
    assert not on_kaggle()


def test_colab_session_deadline() -> None:
    """Colab session deadline is computed cleanly from session start."""
    config = {
        "colab_session": {
            "limit_hours": 12.0,
            "margin_minutes": 30.0,
        }
    }
    deadline = session_deadline(
        config,
        "colab",
        {SESSION_START_ENV: "1000"},
        clock=lambda: 1000.0,
    )
    assert deadline is not None
    assert deadline.stop_at == 1000.0 + (12.0 * 3600) - (30.0 * 60)
    assert not deadline.expired()

    # Expired check
    expired_clock = 1000.0 + (12.0 * 3600) - (20.0 * 60)
    deadline_expired = session_deadline(
        config,
        "colab",
        {SESSION_START_ENV: "1000"},
        clock=lambda: expired_clock,
    )
    assert deadline_expired is not None
    assert deadline_expired.expired()


def test_environment_preflight_packages() -> None:
    """Package check catches missing packages and approves existing ones."""
    preflight = EnvironmentPreflight(
        required_packages=("torch", "non_existent_fake_package_xyz_99")
    )
    issues = preflight.check_packages()
    assert len(issues) == 1
    assert issues[0].category == "packages"
    assert "non_existent_fake_package_xyz_99" in issues[0].message
    assert issues[0].critical is True


def test_environment_preflight_disk_space(tmp_path: Path) -> None:
    """Disk space check verifies available headroom against minimum."""
    preflight = EnvironmentPreflight(min_disk_free_gb=100000.0)  # Unusually large
    free_gb, issues = preflight.check_disk_space(tmp_path)
    assert free_gb >= 0.0
    assert len(issues) == 1
    assert issues[0].category == "disk"
    assert issues[0].critical is True


def test_environment_preflight_hardware_affinity() -> None:
    """Hardware check warns on multi-GPU for small parameter predictors."""
    fake_dev = DeviceInfo(
        device=torch.device("cuda:0"),
        device_type="cuda",
        device_count=2,
        device_names=["Tesla T4", "Tesla T4"],
        total_vram_gb=15.0,
        capability=(7, 5),
        is_cuda=True,
    )
    with patch("training.environment.preflight.discover_device", return_value=fake_dev):
        preflight = EnvironmentPreflight(env="kaggle")
        dev, issues = preflight.check_hardware(model_key="upn_v2")
        assert dev is not None
        assert any(i.category == "hardware" and not i.critical for i in issues)


def test_environment_preflight_dataset_presence(tmp_path: Path) -> None:
    """Dataset presence verifies existence under specified search roots."""
    ds_dir = tmp_path / "MyTestDataset"
    ds_dir.mkdir()

    preflight = EnvironmentPreflight()
    # Found
    issues = preflight.check_dataset_presence("MyTestDataset", [tmp_path])
    assert len(issues) == 0

    # Missing
    missing_issues = preflight.check_dataset_presence("NonExistentDataset", [tmp_path])
    assert len(missing_issues) == 1
    assert missing_issues[0].critical is True
