"""Checkpoint Service for LemGendary Model Training Suite.

Provides inspection, listing, and lifecycle management of saved model checkpoints.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any
import torch

from training.checkpoint.manager import safe_load_checkpoint
from training.utils.paths import get_project_root


class CheckpointService:
    """Manages model checkpoint inspection, listing, and lifecycle pruning."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.checkpoints_root = self.project_root / "checkpoints"

    def list_checkpoints(self, model_key: str | None = None) -> list[dict[str, Any]]:
        """List available checkpoint files on disk.

        Args:
            model_key: Optional specific model key to filter by.

        Returns:
            list[dict[str, Any]]: Metadata list of discovered checkpoints.
        """
        results: list[dict[str, Any]] = []
        if not self.checkpoints_root.exists():
            return results

        target_dirs: list[Path] = []
        if model_key:
            model_dir = self.checkpoints_root / model_key
            if model_dir.exists():
                target_dirs.append(model_dir)
        else:
            target_dirs = [d for d in self.checkpoints_root.iterdir() if d.is_dir()]

        for m_dir in target_dirs:
            key = m_dir.name
            for file_path in m_dir.glob("*.pth"):
                stat = file_path.stat()
                epoch_match = re.search(r"epoch_(\d+)", file_path.name)
                epoch = int(epoch_match.group(1)) if epoch_match else None
                is_best = "best" in file_path.name.lower()

                results.append({
                    "model_key": key,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "epoch": epoch,
                    "is_best": is_best,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

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

        # Determine parameter count and keys
        model_state = loaded.get("model_state") or loaded.get("state_dict") or {}
        param_count = 0
        layer_names: list[str] = []

        if isinstance(model_state, dict):
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
