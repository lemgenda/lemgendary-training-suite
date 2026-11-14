"""Kaggle Hub model uploads (per-epoch auto-sync) and artifact downloads.

Kernel bundles are built by :mod:`training.cloud.kernel_bundle`.
"""

from __future__ import annotations

import importlib
import logging
import shutil
from pathlib import Path
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.manager import CloudSyncError
from training.cloud.sync_manifest import SyncManifest
from training.cloud.variations import ALL_VARIATIONS, PYTORCH_DEFAULT
from training.cloud.version_notes import build_version_notes, release_facts

logger = logging.getLogger("lemtrain.cloud.kaggle_hub")


class KaggleHubManager:
    """Manages Kaggle Model Hub uploads and artifact downloads."""

    def __init__(
        self,
        username: str | None = None,
        key: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        resolved_user, resolved_key = resolve_kaggle_credentials(username, key)
        self.username = resolved_user
        self.key = resolved_key

    def sync(
        self,
        model_name: str,
        epoch: int,
        src_dir: Path,
        is_best: bool = False,
    ) -> bool:
        """Upload model variations of ``src_dir`` to Kaggle Hub.

        When ``is_best`` is True and staged exports exist in ``dist/kaggle``, pushes all
        variations (PyTorch default, PyTorch FP32, ONNX FP16, ONNX FP32) via
        :meth:`KaggleModelSyncManager.push_model`. Otherwise uploads only the isolated
        ``pytorch/default`` payload via :meth:`KaggleModelSyncManager.push_variation`.
        ``src_dir`` is ``<hub root>/<model_name>``.
        """
        publisher = KaggleModelSyncManager(
            username=self.username, key=self.key, hub_root=src_dir.parent
        )
        if not SyncManifest(src_dir).pushed_hashes(PYTORCH_DEFAULT.key):
            # First upload from this folder (a new model or a fresh Kaggle session): set the
            # model page first, so a new model never starts with Kaggle's defaults.
            page = publisher.sync_model_page(model_name)
            if page.get("model") is None:
                raise CloudSyncError(
                    code="KAGGLE_METADATA_FAILED",
                    message=f"Model page sync failed for {model_name}: {page.get('errors')}",
                    retryable=True,
                )

        dist_dir = src_dir / "dist" / "kaggle"
        if is_best and dist_dir.exists() and any(dist_dir.iterdir()):
            res = publisher.push_model(
                model_key=model_name,
                variation=ALL_VARIATIONS,
                version_notes=build_version_notes(
                    model_name,
                    release_facts(src_dir, epoch=epoch),
                    label="all",
                    reason="New SOTA Milestone",
                ),
            )
            if not res.success:
                raise CloudSyncError(
                    code="KAGGLE_UPLOAD_FAILED",
                    message=f"Failed to upload all variations for {model_name} epoch {epoch}: "
                    f"{[v.error for v in res.variations.values() if not v.success]}",
                    retryable=True,
                )
            logger.info("Uploaded all variations for %s epoch %d to Kaggle Hub", model_name, epoch)
            return True

        result = publisher.push_variation(
            model_name,
            PYTORCH_DEFAULT.key,
            version_notes=build_version_notes(
                model_name,
                release_facts(src_dir, epoch=epoch),
                label=PYTORCH_DEFAULT.label,
                reason="Auto-sync",
            ),
        )
        if result.skipped:
            logger.info("Kaggle Hub already holds this payload of %s; no new version", model_name)
            return True
        if not result.success:
            raise CloudSyncError(
                code="KAGGLE_UPLOAD_FAILED",
                message=f"Failed to upload {model_name} epoch {epoch} to Kaggle Hub: "
                f"{result.error}",
                retryable=True,
            )
        logger.info("Uploaded %s epoch %d to %s", model_name, epoch, result.handle)
        return True

    def pull(self, remote_identifier: str, dest_dir: Path) -> bool:
        """Download model artifacts or dataset from Kaggle Hub."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            kagglehub = importlib.import_module("kagglehub")
        except ImportError as exc:
            raise CloudSyncError(
                code="KAGGLEHUB_MISSING",
                message="kagglehub library is not installed",
                retryable=False,
            ) from exc

        try:
            download_path_str = kagglehub.model_download(remote_identifier)
            download_path = Path(download_path_str)
            if download_path.exists():
                if download_path.is_file():
                    shutil.copy2(download_path, dest_dir / download_path.name)
                elif download_path.is_dir():
                    for item in download_path.iterdir():
                        target = dest_dir / item.name
                        if item.is_dir():
                            shutil.copytree(item, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, target)
                logger.info(
                    "Successfully pulled Kaggle artifact %s -> %s", remote_identifier, dest_dir
                )
                return True
            return False
        except kagglehub_errors() as exc:
            logger.error("Kaggle Hub download failed for %s: %s", remote_identifier, exc)
            raise CloudSyncError(
                code="KAGGLE_DOWNLOAD_FAILED",
                message=f"Failed pulling Kaggle artifact {remote_identifier}: {exc}",
                retryable=True,
            ) from exc

    def probe_health(self) -> dict[str, Any]:
        """Verify Kaggle credentials and kagglehub availability."""
        has_sdk = False
        try:
            importlib.import_module("kagglehub")
            has_sdk = True
        except ImportError as exc:
            logger.debug("kagglehub SDK not importable: %s", exc)

        return {
            "provider": "kaggle",
            "username": self.username,
            "has_key": bool(self.key),
            "has_sdk": has_sdk,
        }
