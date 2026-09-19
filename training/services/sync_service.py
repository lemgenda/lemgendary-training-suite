"""Cloud Synchronization Service for LemGendary Model Training Suite.

Coordinates remote storage synchronization with Google Drive, Kaggle Hub, and GitHub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.cloud.gdrive import GDriveSyncManager
from training.cloud.git_hub import GitHubSyncManager
from training.cloud.kaggle_hub import KaggleHubManager
from training.utils.paths import get_project_root


class CloudSyncService:
    """Manages cloud synchronization and provider health checks."""

    SUPPORTED_TARGETS: list[str] = ["gdrive", "kaggle", "github"]

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.gdrive_manager = GDriveSyncManager()
        self.kaggle_manager = KaggleHubManager()
        self.github_manager = GitHubSyncManager()

    def probe_providers(self) -> dict[str, dict[str, Any]]:
        """Probe connectivity and health for all cloud synchronization providers.

        Returns:
            dict[str, dict[str, Any]]: Health status mapped by provider name.
        """
        return {
            "gdrive": self.gdrive_manager.probe_health(),
            "kaggle": self.kaggle_manager.probe_health(),
            "github": self.github_manager.probe_health(),
        }

    def sync_model(
        self,
        model_key: str,
        target: str = "gdrive",
        epoch: int = 1,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Synchronize model checkpoints and artifacts to specified cloud target.

        Args:
            model_key: Model key identifier.
            target: Destination provider ('gdrive', 'kaggle', 'github').
            epoch: Checkpoint epoch index to synchronize.
            dry_run: If True, validate parameters without executing transfer.

        Returns:
            dict[str, Any]: Synchronization summary.
        """
        if target not in self.SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported cloud sync target '{target}'. Supported: {self.SUPPORTED_TARGETS}"
            )

        src_dir = self.project_root / "checkpoints" / model_key
        if not src_dir.exists():
            # Check LemGendaryModels root folder fallback
            fallback_dir = self.project_root.parent / "LemGendaryModels" / model_key
            if fallback_dir.exists():
                src_dir = fallback_dir
            else:
                return {
                    "model_key": model_key,
                    "target": target,
                    "success": False,
                    "message": f"Source checkpoint directory not found: {src_dir}",
                }

        if dry_run:
            return {
                "model_key": model_key,
                "target": target,
                "src_dir": str(src_dir),
                "epoch": epoch,
                "dry_run": True,
                "success": True,
            }

        success = False
        message = ""

        if target == "gdrive":
            success = self.gdrive_manager.sync(model_name=model_key, epoch=epoch, src_dir=src_dir)
            message = "Google Drive synchronization finished."
        elif target == "kaggle":
            success = self.kaggle_manager.sync(model_name=model_key, epoch=epoch, src_dir=src_dir)
            message = "Kaggle Hub synchronization finished."
        elif target == "github":
            success = self.github_manager.sync(model_name=model_key, epoch=epoch, src_dir=src_dir)
            message = "GitHub synchronization finished."

        return {
            "model_key": model_key,
            "target": target,
            "src_dir": str(src_dir),
            "epoch": epoch,
            "success": success,
            "message": message,
        }
