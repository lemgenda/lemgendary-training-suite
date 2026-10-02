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

        # Resolve configured external LemGendaryModels hub directory
        hub_path = (self.project_root / ".." / "LemGendaryModels").resolve()
        cfg_path = self.project_root / "config.yaml"
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                rel = cfg.get("paths", {}).get("checkpoints_root")
                if rel:
                    candidate = (self.project_root / rel).resolve()
                    if candidate.exists():
                        hub_path = candidate
            except Exception:
                pass
        self.hub_root = hub_path if hub_path.exists() else None

    def list_checkpoints(self, model_key: str | None = None) -> list[dict[str, Any]]:
        """List available checkpoint files on disk from local checkpoints and model hub.

        Args:
            model_key: Optional specific model key to filter by.

        Returns:
            list[dict[str, Any]]: Metadata list of discovered checkpoints.
        """
        results: list[dict[str, Any]] = []
        target_dirs: list[tuple[str, Path]] = []

        if self.checkpoints_root.exists():
            if model_key:
                model_dir = self.checkpoints_root / model_key
                if model_dir.exists():
                    target_dirs.append((model_key, model_dir))
            else:
                for d in self.checkpoints_root.iterdir():
                    if d.is_dir() and not d.name.startswith((".", "_")):
                        target_dirs.append((d.name, d))

        if self.hub_root and self.hub_root.exists():
            if model_key:
                hub_dir = self.hub_root / model_key
                if hub_dir.exists():
                    target_dirs.append((model_key, hub_dir))
            else:
                for d in self.hub_root.iterdir():
                    if d.is_dir() and not d.name.startswith((".", "_")):
                        target_dirs.append((d.name, d))

        seen_paths: set[str] = set()

        for key, m_dir in target_dirs:
            # Check for metrics.csv to determine recorded epochs
            max_epoch_from_csv: int | None = None
            csv_path = m_dir / "metrics.csv"
            if csv_path.exists():
                try:
                    import csv
                    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            ep_val = row.get("Epoch") or row.get("epoch")
                            if ep_val:
                                ep_num = int(ep_val)
                                if max_epoch_from_csv is None or ep_num > max_epoch_from_csv:
                                    max_epoch_from_csv = ep_num
                except Exception:
                    pass

            candidates = (
                list(m_dir.glob("*.pth"))
                + list((m_dir / "checkpoints").glob("*.pth"))
                + list(m_dir.glob("*.pt"))
            )

            for file_path in candidates:
                resolved_str = str(file_path.resolve())
                if resolved_str in seen_paths:
                    continue
                seen_paths.add(resolved_str)

                stat = file_path.stat()
                epoch_match = re.search(r"epoch_(\d+)", file_path.name)
                epoch = int(epoch_match.group(1)) if epoch_match else max_epoch_from_csv
                is_best = "best" in file_path.name.lower() or file_path.suffix == ".pt"

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
        if best_loss is None and "best_fitness" in loaded:
            best_loss = loaded.get("best_fitness")
        if not metrics and "best_fitness" in loaded:
            metrics = {"fitness": float(loaded.get("best_fitness", 0.0))}

        # Determine parameter count and keys
        model_state = loaded.get("model_state") or loaded.get("state_dict") or {}
        param_count = 0
        layer_names: list[str] = []

        if isinstance(model_state, dict) and model_state:
            for name, tensor in model_state.items():
                layer_names.append(str(name))
                if isinstance(tensor, torch.Tensor):
                    param_count += tensor.numel()
        elif "model" in loaded and hasattr(loaded["model"], "named_parameters"):
            try:
                for name, param in loaded["model"].named_parameters():
                    layer_names.append(str(name))
                    if isinstance(param, torch.Tensor):
                        param_count += param.numel()
            except Exception:
                pass

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
