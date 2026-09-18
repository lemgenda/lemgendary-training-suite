"""Atomic checkpoint save, load, and disk headroom auditing."""

import logging
import os
from pathlib import Path
import shutil
from typing import Any
import torch

logger = logging.getLogger("lemtrain.checkpoint.manager")


class CheckpointSaveError(RuntimeError):
    """Raised when an atomic checkpoint save cannot be completed."""
    pass


def audit_disk_space(target_dir: Path, min_free_gb: float = 5.0) -> float:
    """Audit available disk space in GB, triggering emergency pruning of .tmp files if critical."""
    try:
        _, _, free_bytes = shutil.disk_usage(str(target_dir))
        free_gb = free_bytes / (1024**3)
    except OSError as exc:
        logger.warning("Failed querying disk usage for %s: %s", target_dir, exc)
        return 999.0

    if free_gb < 1.0:
        logger.warning("Low disk space on %s: %.2f GB available. Pruning stale temporary files...", target_dir, free_gb)
        try:
            for item in target_dir.iterdir():
                if item.is_file() and (item.suffix == ".tmp" or "_progress.pth" in item.name):
                    try:
                        item.unlink()
                        logger.info("Pruned stale temporary file: %s", item.name)
                    except OSError as unlink_err:
                        logger.debug("Could not remove %s: %s", item, unlink_err)
            _, _, free_bytes = shutil.disk_usage(str(target_dir))
            free_gb = free_bytes / (1024**3)
        except OSError as list_err:
            logger.debug("Could not list %s for pruning: %s", target_dir, list_err)

    if free_gb < 0.2:
        raise CheckpointSaveError(
            f"Disk critically full on {target_dir} ({free_gb:.2f} GB remaining). Save aborted to preserve filesystem."
        )

    if free_gb < min_free_gb:
        logger.warning("Disk space warning on %s: %.2f GB remaining (< %.1f GB threshold).", target_dir, free_gb, min_free_gb)

    return free_gb


def safe_atomic_save(payload: Any, path: Path | str, min_free_gb: float = 5.0) -> bool:
    """Save an object to disk atomically via a temporary file with disk headroom checking."""
    target_path = Path(path).resolve()
    parent_dir = target_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    try:
        audit_disk_space(parent_dir, min_free_gb=min_free_gb)
    except CheckpointSaveError as exc:
        logger.critical("Disk check failed during checkpoint save: %s", exc)
        return False

    tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    try:
        torch.save(payload, str(tmp_path))
        # Atomic replace on POSIX, atomic on Windows with os.replace
        os.replace(str(tmp_path), str(target_path))
        logger.info("Saved atomic checkpoint to %s", target_path)
        return True
    except Exception as exc:
        logger.error("Failed saving atomic checkpoint to %s: %s", target_path, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as unlink_err:
                logger.debug("Failed cleaning up tmp checkpoint %s: %s", tmp_path, unlink_err)
        return False


def safe_load_checkpoint(path: Path | str, map_location: Any = "cpu") -> dict[str, Any]:
    """Safely load a PyTorch checkpoint file."""
    ckpt_path = Path(path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

    if ckpt_path.stat().st_size == 0:
        raise ValueError(f"Checkpoint file is empty (0 bytes): {ckpt_path}")

    try:
        # Load with weights_only=False for complete state recovery
        payload = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older torch versions lacking weights_only kwarg
        payload = torch.load(str(ckpt_path), map_location=map_location)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload from checkpoint {ckpt_path}, got {type(payload).__name__}")

    return payload
