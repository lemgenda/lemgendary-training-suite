"""Checkpoint synchronization CLI utility."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil

from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.checkpoint_sync")


def sync_checkpoints(model_name: str, target_dir: str = "/kaggle/working/export") -> None:
    """Synchronize metrics and checkpoints into export persistence directory."""
    project_root = get_project_root()
    formatted_name = f"Lemgendary_{model_name.replace('_', ' ').title().replace(' ', '_')}_Checkpoints"
    persistence_root = Path(target_dir) / formatted_name
    persistence_root.mkdir(parents=True, exist_ok=True)

    hub_model_dir = (project_root.parent / "LemGendaryModels" / model_name).resolve()

    # 1. Sync metrics.csv
    src_metrics = hub_model_dir / "metrics.csv"
    if not src_metrics.exists():
        src_metrics = project_root / "metrics.csv"
    if src_metrics.exists():
        shutil.copy2(src_metrics, persistence_root / "metrics.csv")
        logger.info("Synced metrics.csv -> %s", persistence_root)

    # 2. Sync Checkpoints
    src_ckpt_dir = hub_model_dir / "checkpoints"
    if not src_ckpt_dir.exists():
        src_ckpt_dir = project_root / "checkpoints"
    dst_ckpt_dir = persistence_root / "checkpoints"
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if src_ckpt_dir.exists():
        for f in src_ckpt_dir.iterdir():
            if f.suffix == ".pth" and (model_name in f.name or len(list(src_ckpt_dir.glob("*.pth"))) <= 10):
                shutil.copy2(f, dst_ckpt_dir / f.name)
                logger.info("Synced %s -> %s", f.name, dst_ckpt_dir)


def main() -> None:
    """CLI entrypoint for checkpoint synchronization."""
    parser = argparse.ArgumentParser(description="LemGendary Checkpoint Sync")
    parser.add_argument("--model", type=str, required=True, help="Model name identifier")
    parser.add_argument("--target", type=str, default="/kaggle/working/export", help="Destination export directory")
    args = parser.parse_args()

    sync_checkpoints(args.model, args.target)


if __name__ == "__main__":
    main()
