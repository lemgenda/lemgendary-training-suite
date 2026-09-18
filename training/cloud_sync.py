"""Legacy cloud synchronizer shim delegating to training.cloud."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any

from training.cloud.gdrive import GDriveSyncManager
from training.cloud.git_hub import GitHubSyncManager
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.manager import CloudSyncError

logger = logging.getLogger("lemtrain.cloud_sync")

_sync_lock = threading.Lock()
_active_sync_thread: threading.Thread | None = None


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

        self.github_mgr = GitHubSyncManager()
        self.kaggle_mgr = KaggleHubManager(config=self.config)
        self.gdrive_mgr = GDriveSyncManager()

        self.hub_root = self.github_mgr.hub_root
        self.model_dir = self.hub_root / self.model_name
        self.checkpoint_dir = self.model_dir / "checkpoints"

    def sync(self) -> None:
        """Coordinate multi-target cloud synchronization."""
        logger.info("Starting Cloud Synchronization Phase (Epoch %d)...", self.epoch)
        is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_WORKING_DIR"))

        if is_kaggle:
            logger.info("Operating in Kaggle-Native mode. GitHub sync bypassed.")
            kaggle_success = self._sync_to_kaggle()
            if not self.is_mid_epoch and kaggle_success:
                logger.info("Epoch %d finished and Kaggle version confirmed. Syncing to Google Drive...", self.epoch)
                self._sync_to_gdrive()
        else:
            logger.info("Operating in Hybrid mode. Syncing to GitHub, Kaggle, and Google Drive.")
            self._sync_to_github()
            self._sync_to_kaggle()
            self._sync_to_gdrive()

    def _sync_to_github(self) -> bool:
        """Delegate to GitHubSyncManager."""
        try:
            return self.github_mgr.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
            logger.warning("GitHub sync notice: %s", exc)
            return False

    def _sync_to_kaggle(self) -> bool:
        """Delegate to KaggleHubManager."""
        if not self.checkpoint_dir.exists():
            return False
        try:
            return self.kaggle_mgr.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
            logger.warning("Kaggle Hub sync notice: %s", exc)
            return False

    def _sync_to_gdrive(self) -> bool:
        """Delegate to GDriveSyncManager."""
        fleet_cfg = self.config.get("fleet", {}) if isinstance(self.config, dict) else {}
        if not fleet_cfg.get("google_drive_sync", True):
            return False
        try:
            return self.gdrive_mgr.sync(self.model_name, self.epoch, self.model_dir)
        except CloudSyncError as exc:
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
    global _active_sync_thread
    manager = CloudSyncManager(model_name, epoch, config, is_mid_epoch=is_mid_epoch)

    if wait:
        with _sync_lock:
            manager.sync()
    else:
        def _worker() -> None:
            with _sync_lock:
                manager.sync()

        t = threading.Thread(target=_worker, daemon=False)
        _active_sync_thread = t
        t.start()
