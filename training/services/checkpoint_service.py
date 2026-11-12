"""Checkpoint Service for LemGendary Model Training Suite.

Provides inspection, listing, and lifecycle management of saved model checkpoints.
"""

from __future__ import annotations

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from training.checkpoint.manager import safe_load_checkpoint
from training.utils.model_payload import ModelPayloadError, model_payload_of
from training.utils.paths import get_models_hub_root, get_project_root

logger = logging.getLogger("lemtrain.services.checkpoint_service")


def _model_dirs(root: Path, model_key: str | None) -> list[tuple[str, Path]]:
    """``(key, dir)`` of ``model_key`` below ``root``, or of every visible model folder."""
    if model_key:
        model_dir = root / model_key
        return [(model_key, model_dir)] if model_dir.exists() else []
    return [(d.name, d) for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))]


def _max_recorded_epoch(csv_path: Path) -> int | None:
    """Highest ``Epoch``/``epoch`` in metrics.csv (rows read before a bad cell count)."""
    max_epoch_from_csv: int | None = None
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep_val = row.get("Epoch") or row.get("epoch")
                    if ep_val:
                        ep_num = int(ep_val)
                        if max_epoch_from_csv is None or ep_num > max_epoch_from_csv:
                            max_epoch_from_csv = ep_num
        except (OSError, csv.Error, ValueError) as csv_err:
            logger.warning("Could not parse recorded epochs from %s: %s", csv_path, csv_err)
    return max_epoch_from_csv


def _checkpoint_files(m_dir: Path) -> list[Path]:
    """Checkpoint candidates of a model folder, in listing order (duplicates possible)."""
    return (
        list(m_dir.glob("*.pth"))
        + list((m_dir / "checkpoints").glob("*.pth"))
        + list((m_dir / "checkpoints").glob("*.pt"))
        + list((m_dir / "dist" / "kaggle" / "pytorch").glob("*.pt"))
        + list(m_dir.glob("*.pt"))
    )


def _checkpoint_entry(key: str, file_path: Path, max_epoch_from_csv: int | None) -> dict[str, Any]:
    """Listing entry of one checkpoint file (epoch from the name, else from metrics.csv)."""
    stat = file_path.stat()
    epoch_match = re.search(r"epoch_(\d+)", file_path.name)
    epoch = int(epoch_match.group(1)) if epoch_match else max_epoch_from_csv
    is_best = "best" in file_path.name.lower() or file_path.suffix == ".pt"
    return {
        "model_key": key,
        "filename": file_path.name,
        "path": str(file_path),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "epoch": epoch,
        "is_best": is_best,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


class CheckpointService:
    """Manages model checkpoint inspection, listing, and lifecycle pruning."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.checkpoints_root = self.project_root / "checkpoints"

        # Resolve configured external LemGendaryModels hub directory
        hub_path = get_models_hub_root(self.project_root)
        cfg_path = self.project_root / "config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                rel = cfg.get("paths", {}).get("checkpoints_root")
                if rel:
                    candidate = (self.project_root / rel).resolve()
                    if candidate.exists():
                        hub_path = candidate
            except (OSError, yaml.YAMLError, AttributeError, TypeError) as cfg_err:
                logger.warning(
                    "Could not read checkpoints_root from %s; using default hub %s: %s",
                    cfg_path,
                    hub_path,
                    cfg_err,
                )
        self.hub_root = hub_path if hub_path.exists() else None

    def list_checkpoints(self, model_key: str | None = None) -> list[dict[str, Any]]:
        """List available checkpoint files on disk from local checkpoints and model hub.

        Args:
            model_key: Optional specific model key to filter by.

        Returns:
            list[dict[str, Any]]: Metadata list of discovered checkpoints.
        """
        target_dirs: list[tuple[str, Path]] = []
        if self.checkpoints_root.exists():
            target_dirs.extend(_model_dirs(self.checkpoints_root, model_key))
        if self.hub_root and self.hub_root.exists():
            target_dirs.extend(_model_dirs(self.hub_root, model_key))

        results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for key, m_dir in target_dirs:
            # Check for metrics.csv to determine recorded epochs
            max_epoch_from_csv = _max_recorded_epoch(m_dir / "metrics.csv")
            for file_path in _checkpoint_files(m_dir):
                resolved_str = str(file_path.resolve())
                if resolved_str in seen_paths:
                    continue
                seen_paths.add(resolved_str)
                results.append(_checkpoint_entry(key, file_path, max_epoch_from_csv))

        return sorted(results, key=lambda x: (x["model_key"], x["filename"]))

    def inspect_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Inspect contents, metadata, and parameter shapes of a checkpoint.

        Args:
            checkpoint_path: Path to target checkpoint file.

        Returns:
            dict[str, Any]: Detailed inspection dictionary.
        """
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        loaded = safe_load_checkpoint(path, map_location="cpu")
        if loaded is None:
            raise ValueError(f"Failed to load checkpoint file: {path}")

        stat = path.stat()
        epoch = loaded.get("epoch")
        metrics = loaded.get("metrics", {})
        best_loss = loaded.get("best_loss")
        if best_loss is None and "best_fitness" in loaded:
            best_loss = loaded.get("best_fitness")
        if not metrics and "best_fitness" in loaded:
            metrics = {"fitness": float(loaded.get("best_fitness", 0.0))}

        # Determine parameter count and keys (release payload, checkpoint or bare state dict)
        param_count = 0
        layer_names: list[str] = []
        try:
            model_state = model_payload_of(loaded).state_dict
        except ModelPayloadError as state_err:
            logger.warning("No model weights found in %s: %s", path, state_err)
            model_state = {}
        for name, tensor in model_state.items():
            layer_names.append(str(name))
            if isinstance(tensor, torch.Tensor):
                param_count += tensor.numel()

        return {
            "path": str(path),
            "filename": path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "epoch": epoch,
            "best_loss": best_loss,
            "metrics": metrics,
            "parameter_count": param_count,
            "layers_count": len(layer_names),
            "keys": list(loaded.keys()),
        }

    def prune_checkpoints(
        self,
        model_key: str,
        keep_last_k: int = 3,
        keep_best: bool = True,
    ) -> list[str]:
        """Prune older intermediate checkpoints while retaining best and recent k checkpoints.

        Args:
            model_key: Target model key.
            keep_last_k: Number of most recent numbered epoch checkpoints to retain.
            keep_best: Retain best checkpoint regardless of epoch.

        Returns:
            list[str]: Filenames of pruned (deleted) checkpoints.
        """
        model_dir = self.checkpoints_root / model_key
        if not model_dir.exists():
            return []

        # Gather epoch checkpoints
        epoch_files: list[tuple[int, Path]] = []
        for file_path in model_dir.glob("*.pth"):
            if "best" in file_path.name.lower() and keep_best:
                continue
            if file_path.name == "progress.pth":
                continue
            match = re.search(r"epoch_(\d+)", file_path.name)
            if match:
                epoch_files.append((int(match.group(1)), file_path))

        epoch_files.sort(key=lambda x: x[0])
        pruned: list[str] = []

        if len(epoch_files) > keep_last_k:
            to_delete = epoch_files[:-keep_last_k]
            for _, path in to_delete:
                pruned.append(path.name)
                path.unlink(missing_ok=True)

        return pruned
