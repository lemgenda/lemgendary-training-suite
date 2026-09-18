"""Kaggle Hub model, dataset, and kernel execution manager."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.manager import CloudManager, CloudSyncError
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.cloud.kaggle_hub")


class KaggleHubManager:
    """Manages Kaggle Model Hub uploads, artifact downloads, and headless kernel bundles."""

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

    def get_handle(self, model_name: str) -> str:
        """Construct standard Kaggle model handle: <user>/lemgendary-<model>-checkpoints/pytorch/default."""
        model_slug = model_name.replace("_", "-")
        if "nima-aesthetic" in model_slug:
            model_slug = model_slug.replace("nima-aesthetic", "nima-aesthetics")

        slug_prefix = self.config.get("kaggle_slug_prefix", "lemgendary-")
        slug_suffix = self.config.get("kaggle_slug_suffix", "-checkpoints")
        return f"{self.username}/{slug_prefix}{model_slug}{slug_suffix}/pytorch/default"

    def get_kernel_slug(self, model_name: str) -> str:
        """Generate standardized Kaggle kernel slug."""
        clean_model = model_name.replace("_", "-")
        return f"{self.username}/lemgendary-{clean_model}-training"

    def create_cloud_kernel_bundle(self, model_name: str, gpu: str = "T4") -> Path:
        """Package a standalone Kaggle kernel bundle containing metadata and execution script."""
        project_root = get_project_root()
        kernel_dir = project_root / "cloud_jobs" / f"{model_name}_cloud_job"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        slug = f"lemgendary-{model_name.replace('_', '-')}-training"
        title = f"LemGendary {model_name.replace('_', ' ').title()} Training"

        gpu_shape_map = {
            "T4": "NvidiaTeslaT4",
            "P100": "NvidiaTeslaP100",
            "V100": "NvidiaTeslaV100",
        }
        accelerator = gpu_shape_map.get(gpu.upper(), "NvidiaTeslaT4")

        metadata = {
            "id": f"{self.username}/{slug}",
            "title": title,
            "code_file": "train_cloud.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "accelerator": accelerator,
        }

        import json
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        script_content = f'''"""Automated Headless Kaggle Cloud Training Job for {model_name}."""
import os
import subprocess
import sys

print("Initializing LemGendary Cloud Worker for {model_name}...")
res = subprocess.run([sys.executable, "-m", "training.core_loop", "--model", "{model_name}"], check=False)
sys.exit(res.returncode)
'''
        (kernel_dir / "train_cloud.py").write_text(script_content, encoding="utf-8")
        logger.info("Created Kaggle cloud kernel bundle at %s", kernel_dir)
        return kernel_dir

    def sync(
        self,
        model_name: str,
        epoch: int,
        src_dir: Path,
        **kwargs: Any,
    ) -> bool:
        """Stage artifacts and upload model version to Kaggle Model Hub."""
        handle = self.get_handle(model_name)
        staging_dir = Path("/kaggle/working/.kaggle_upload_stage") if os.name != "nt" else (src_dir / ".kaggle_upload_stage")

        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)

            # Copy or hardlink contents
            for item in src_dir.iterdir():
                dst = staging_dir / item.name
                if item.name.lower() == "checkpoints" and item.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in item.iterdir():
                        if f.is_file():
                            try:
                                os.link(f, dst / f.name)
                            except OSError:
                                shutil.copy2(f, dst / f.name)
                elif item.name != ".kaggle_upload_stage":
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    elif item.is_file():
                        shutil.copy2(item, dst)

            # Dynamic import of kagglehub
            try:
                kagglehub = importlib.import_module("kagglehub")
            except ImportError as exc:
                raise CloudSyncError(
                    code="KAGGLEHUB_MISSING",
                    message="kagglehub library is not installed",
                    retryable=False,
                ) from exc

            logger.info("Uploading model to Kaggle Hub (%s)...", handle)
            kagglehub.model_upload(
                handle=handle,
                local_model_dir=str(staging_dir),
                version_notes=f"SOTA Update: {model_name} | Epoch {epoch}",
            )
            logger.info("Successfully uploaded %s to Kaggle Hub.", handle)
            return True
        except CloudSyncError:
            raise
        except Exception as exc:
            raise CloudSyncError(
                code="KAGGLE_UPLOAD_FAILED",
                message=f"Failed to upload {model_name} to Kaggle Hub: {exc}",
                retryable=True,
            ) from exc
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def pull(self, remote_identifier: str, dest_dir: Path, **kwargs: Any) -> bool:
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
                logger.info("Successfully pulled Kaggle artifact %s -> %s", remote_identifier, dest_dir)
                return True
            return False
        except Exception as exc:
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
        except ImportError:
            pass

        return {
            "provider": "kaggle",
            "username": self.username,
            "has_key": bool(self.key),
            "has_sdk": has_sdk,
        }
