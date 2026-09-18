"""Google Drive cloud manager legacy shim delegating to training.cloud."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

import yaml

from training.cloud.credentials import resolve_gdrive_credentials
from training.cloud.gdrive import DEFAULT_ROOT_FOLDER_ID, GDriveSyncManager
from training.cloud.manager import CloudSyncError

logger = logging.getLogger("lemtrain.gdrive_cloud_manager")


def resolve_mounted_drive_root() -> Path | None:
    """Detect local or Colab mounted Google Drive root directory."""
    return GDriveSyncManager.resolve_mounted_root()


class GDriveCloudManager:
    """Legacy wrapper for Google Drive cloud synchronization."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        folder_id: str | None = None,
        token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = config or {}
        fleet_cfg = self.config.get("fleet", {})
        self.folder_id = folder_id or fleet_cfg.get("google_drive_folder_id", DEFAULT_ROOT_FOLDER_ID)
        self.token = resolve_gdrive_credentials(token)
        self.suite_dir = Path(__file__).resolve().parent.parent

        self.gdrive_mgr = GDriveSyncManager(
            root_folder_id=self.folder_id,
            token=self.token,
        )
        self.model_dir = self._resolve_model_dir()
        self.checkpoint_dir = self.model_dir / "checkpoints"

    def _resolve_model_dir(self) -> Path:
        candidates = [
            Path("/kaggle/working/LemGendaryModels") / self.model_name,
            Path("/content/LemGendaryModels") / self.model_name,
            self.suite_dir.parent / "LemGendaryModels" / self.model_name,
            self.suite_dir / "hub" / self.model_name,
        ]
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return cand.resolve()
        return (self.suite_dir.parent / "LemGendaryModels" / self.model_name).resolve()

    def discover_artifacts(self) -> list[tuple[Path, str]]:
        """Discover valid artifacts for current model."""
        if not self.model_dir.exists():
            return []

        artifacts: list[tuple[Path, str]] = []
        if self.checkpoint_dir.exists():
            for pth_file in self.checkpoint_dir.glob("*.pth"):
                artifacts.append((pth_file, f"checkpoints/{pth_file.name}"))

        for item in self.model_dir.iterdir():
            if item.is_file() and item.suffix.lower() in [".csv", ".md", ".onnx", ".pt", ".png"]:
                artifacts.append((item, item.name))

        return artifacts

    def sync(self) -> bool:
        """Perform synchronization to Google Drive target."""
        if not self.model_dir.exists():
            logger.warning("Local model directory does not exist: %s", self.model_dir)
            return False

        try:
            return self.gdrive_mgr.sync(self.model_name, 0, self.model_dir)
        except CloudSyncError as exc:
            logger.warning("Google Drive sync failed: %s", exc)
            return False


def sync_single_model(model_name: str, config: dict[str, Any], dry_run: bool = False) -> bool:
    """Synchronize single model manifold to Google Drive."""
    mgr = GDriveCloudManager(model_name, config=config)
    if dry_run:
        logger.info("[DRY RUN] Discovered %d artifacts for %s", len(mgr.discover_artifacts()), model_name)
        return True
    return mgr.sync()


def sync_all_models(config: dict[str, Any], dry_run: bool = False) -> dict[str, bool]:
    """Synchronize all models defined in models section of config."""
    models_dict = config.get("models", {})
    results: dict[str, bool] = {}
    for model_name in models_dict:
        results[model_name] = sync_single_model(model_name, config, dry_run=dry_run)
    return results


def main() -> None:
    """CLI entrypoint for Google Drive model synchronization."""
    parser = argparse.ArgumentParser(description="LemGendary Google Drive Model Synchronizer")
    parser.add_argument("--model", type=str, help="Specific model key to synchronize")
    parser.add_argument("--all", action="store_true", help="Synchronize all models across the fleet")
    parser.add_argument("--dry-run", action="store_true", help="Preview synchronization without file transfers")
    parser.add_argument("--folder-id", type=str, default=DEFAULT_ROOT_FOLDER_ID, help="Google Drive Root Folder ID")
    parser.add_argument("--token", type=str, help="Explicit Google Drive authorization token")
    args = parser.parse_args()

    suite_dir = Path(__file__).resolve().parent.parent
    config_path = suite_dir / "config.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Could not read config.yaml: %s", e)

    if args.folder_id:
        config.setdefault("fleet", {})["google_drive_folder_id"] = args.folder_id

    if args.model:
        success = sync_single_model(args.model, config, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    elif args.all:
        results = sync_all_models(config, dry_run=args.dry_run)
        any_failed = any(not v for v in results.values())
        sys.exit(1 if any_failed else 0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
