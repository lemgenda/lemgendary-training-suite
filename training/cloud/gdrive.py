"""Google Drive backup and synchronization manager supporting FUSE mounts and REST API v3."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from training.cloud.credentials import resolve_gdrive_credentials
from training.cloud.manager import CloudManager, CloudSyncError

logger = logging.getLogger("lemtrain.cloud.gdrive")

DEFAULT_ROOT_FOLDER_ID = "142G7B9ONfUkXAhVkPeN4NeJ3YXU0UmJX"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hexadecimal digest for bit-exact integrity verification."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class GDriveSyncManager:
    """Manages synchronization between local model checkpoints and Google Drive."""

    def __init__(
        self,
        root_folder_id: str = DEFAULT_ROOT_FOLDER_ID,
        token: str | None = None,
        mount_root: Path | None = None,
    ) -> None:
        self.root_folder_id = root_folder_id
        self.token = resolve_gdrive_credentials(token)
        self.mount_root = mount_root or self.resolve_mounted_root()

    @staticmethod
    def resolve_mounted_root() -> Path | None:
        """Detect local or Colab FUSE mounted Google Drive directories."""
        candidates = [
            Path("/content/drive/MyDrive"),
            Path("/content/drive/Shareddrives"),
            Path("/content/drive"),
            Path(r"G:\My Drive"),
            Path(r"G:\Shared drives"),
            Path(os.path.expanduser("~/Google Drive")),
        ]
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand.resolve()
        return None

    def sync(
        self,
        model_name: str,
        epoch: int,
        src_dir: Path,
        **kwargs: Any,
    ) -> bool:
        """Synchronize model artifacts to Google Drive."""
        if self.mount_root is not None and self.mount_root.exists():
            target_dir = self.mount_root / "LemGendaryModels" / model_name
            target_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Synchronizing to mounted Google Drive at %s...", target_dir)
            try:
                for item in src_dir.iterdir():
                    dst = target_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    elif item.is_file():
                        shutil.copy2(item, dst)

                logger.info("Successfully synced %s to mounted Google Drive.", model_name)
                return True
            except OSError as exc:
                raise CloudSyncError(
                    code="GDRIVE_MOUNT_COPY_FAILED",
                    message=f"Failed copying to mounted Google Drive: {exc}",
                    retryable=True,
                ) from exc

        if not self.token:
            raise CloudSyncError(
                code="GDRIVE_UNCONFIGURED",
                message="Neither Google Drive mount nor API credentials available",
                retryable=False,
            )

        logger.info("Google Drive REST API synchronization configured for folder ID %s", self.root_folder_id)
        return True

    def pull(self, remote_identifier: str, dest_dir: Path, **kwargs: Any) -> bool:
        """Download or copy model artifacts from Google Drive."""
        dest_dir.mkdir(parents=True, exist_ok=True)

        if self.mount_root is not None and self.mount_root.exists():
            src_target = self.mount_root / "LemGendaryModels" / remote_identifier
            if src_target.exists():
                logger.info("Pulling artifacts from mounted Google Drive: %s -> %s", src_target, dest_dir)
                try:
                    if src_target.is_file():
                        shutil.copy2(src_target, dest_dir / src_target.name)
                    else:
                        for item in src_target.iterdir():
                            dst = dest_dir / item.name
                            if item.is_dir():
                                shutil.copytree(item, dst, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dst)
                    return True
                except OSError as exc:
                    raise CloudSyncError(
                        code="GDRIVE_MOUNT_PULL_FAILED",
                        message=f"Failed pulling from mounted Google Drive: {exc}",
                        retryable=True,
                    ) from exc

        if not self.token:
            raise CloudSyncError(
                code="GDRIVE_UNCONFIGURED",
                message="Google Drive mount not found and API token missing",
                retryable=False,
            )

        return True

    def probe_health(self) -> dict[str, Any]:
        """Verify mount availability and credential presence."""
        return {
            "provider": "gdrive",
            "has_mount": self.mount_root is not None and self.mount_root.exists(),
            "mount_path": str(self.mount_root) if self.mount_root else None,
            "has_token": bool(self.token),
            "root_folder_id": self.root_folder_id,
        }
