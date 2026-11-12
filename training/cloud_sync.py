"""Legacy cloud synchronizer shim delegating to training.cloud."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from training.cloud.gdrive import GDriveSyncManager
from training.cloud.git_hub import GitHubSyncManager
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.manager import CloudSyncError
from training.cloud.sync_threads import SYNC_THREADS, exit_timeout

logger = logging.getLogger("lemtrain.cloud_sync")

_sync_lock = threading.Lock()


@dataclass
class SyncTargets:
    """The three sync backends, created in GitHub, Kaggle, Google Drive order."""

    config: dict[str, Any]
    github: GitHubSyncManager = field(init=False)
    kaggle: KaggleHubManager = field(init=False)
    gdrive: GDriveSyncManager = field(init=False)

    def __post_init__(self) -> None:
        self.github = GitHubSyncManager()
        self.kaggle = KaggleHubManager(config=self.config)
        self.gdrive = GDriveSyncManager()


class CloudSyncManager:
    """Legacy interface coordinating GitHub, Kaggle, and Google Drive synchronization."""

    def __init__(
        self,
        model_name: str,
        epoch: int,
        config: dict[str, Any] | None = None,
        is_mid_epoch: bool = False,
    ) -> None:
        self.model_name = model_name
        self.epoch = epoch
        self.config = config or {}
        self.is_mid_epoch = is_mid_epoch

        self.targets = SyncTargets(self.config)
        self.model_dir = self.targets.github.hub_root / self.model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"

    def sync(self) -> None:
        """Coordinate multi-target cloud synchronization."""
        logger.info("Starting Cloud Synchronization Phase (Epoch %d)...", self.epoch)
        is_kaggle = bool(
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_WORKING_DIR")
        )

        if is_kaggle:
            logger.info("Operating in Kaggle-Native mode. GitHub sync bypassed.")
            kaggle_success = self.sync_to_kaggle()
            if not self.is_mid_epoch and kaggle_success:
                logger.info(
                    "Epoch %d finished and Kaggle version confirmed. Syncing to Google Drive...",
                    self.epoch,
                )
                self.sync_to_gdrive()
        else:
            logger.info("Operating in Hybrid mode. Syncing to GitHub, Kaggle, and Google Drive.")
            self.sync_to_github()
            self.sync_to_kaggle()
            self.sync_to_gdrive()

    def sync_to_github(self) -> bool:
        """Delegate to GitHubSyncManager."""
        try:
            return self.targets.github.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
            logger.warning("GitHub sync notice: %s", exc)
            return False

    def sync_to_kaggle(self) -> bool:
        """Delegate to KaggleHubManager."""
        if not self.checkpoint_dir.exists():
            return False
        try:
            return self.targets.kaggle.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
            logger.warning("Kaggle Hub sync notice: %s", exc)
            return False

    def sync_to_gdrive(self) -> bool:
        """Delegate to GDriveSyncManager."""
        fleet_cfg = self.config.get("fleet", {}) if isinstance(self.config, dict) else {}
        if not fleet_cfg.get("google_drive_sync", True):
            return False
        try:
            return self.targets.gdrive.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
            if getattr(exc, "code", "") == "GDRIVE_UNCONFIGURED":
                logger.info("Google Drive sync skipped: %s", exc)
            else:
                logger.warning("Google Drive sync notice: %s", exc)
            return False


def trigger_cloud_sync(
    model_name: str,
    epoch: int,
    config: dict[str, Any] | None = None,
    wait: bool = False,
    is_mid_epoch: bool = False,
) -> None:
    """Entry point for training loop to trigger background or synchronous sync."""
    # Under DDP, gate cloud synchronization to Rank 0 to prevent concurrent upload collisions
    if os.environ.get("RANK", "0") != "0":
        logger.debug("Skipping cloud sync on non-zero rank (%s)", os.environ.get("RANK"))
        return

    manager = CloudSyncManager(model_name, epoch, config, is_mid_epoch=is_mid_epoch)

    if wait:
        with _sync_lock:
            manager.sync()
    else:

        def _worker() -> None:
            with _sync_lock:
                manager.sync()

        # Daemon thread with a bounded, logged wait at exit (training/cloud/sync_threads.py).
        SYNC_THREADS.start(_worker, f"cloud-sync-{model_name}-e{epoch}", exit_timeout(config))
