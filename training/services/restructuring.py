"""Model Topology Restructuring Service for LemGendary AI Training Suite.

Reorganizes LemGendaryModels/<model_key>/ into the canonical topology:
- README.md (Root)
- model-metadata.json (Root)
- metrics.csv (Root)
- checkpoints/
- notebooks/
- visualizations/
- dist/kaggle/
  - pytorch/
  - onnx_fp32/
  - onnx_fp16/
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any

from training.checkpoint.lifecycle_manager import canonicalize_checkpoints
from training.cloud.metadata_generator import KaggleMetadataService
from training.doc_generator import ModelDocumentationService
from training.export.staging import VariationStagingService
from training.hub_readme_generator import generate_hub_readme
from training.registry import get_model_registry
from training.utils.fs import remove_tree
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.services.restructuring")


def _relocate_notebooks(model_dir: Path, notebooks_dir: Path) -> list[str]:
    """Move root notebooks into notebooks/ (root duplicates are deleted)."""
    actions: list[str] = []
    for p in model_dir.glob("*.ipynb"):
        dest = notebooks_dir / p.name
        if dest.exists():
            p.unlink()
            actions.append(f"Purged duplicate notebook from root: {p.name}")
        else:
            shutil.move(str(p), str(dest))
            actions.append(f"Moved notebook {p.name} -> notebooks/")
    return actions


def _relocate_visuals(model_dir: Path, visualizations_dir: Path) -> list[str]:
    """Move root images into visualizations/."""
    actions: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        for p in model_dir.glob(ext):
            dest = visualizations_dir / p.name
            if dest.exists() and dest != p:
                p.unlink()
            else:
                shutil.move(str(p), str(dest))
            actions.append(f"Moved visual asset {p.name} -> visualizations/")
    return actions


def _remove_rogue_training_dir(model_dir: Path) -> list[str]:
    """Clean up rogue training/ directory if present."""
    actions: list[str] = []
    training_dir = model_dir / "training"
    if training_dir.is_dir():
        if remove_tree(training_dir, purpose="rogue training/ cleanup"):
            actions.append("Removed rogue training/ directory")
        else:
            actions.append("Could not remove rogue training/ directory")
    return actions


def _purge_root_binaries(model_dir: Path) -> list[str]:
    """Ensure no binaries remain in the root directory."""
    actions: list[str] = []
    for ext in ("*.pt", "*.onnx", "*.data", "*.pth"):
        for leftover in model_dir.glob(ext):
            if leftover.is_file():
                leftover.unlink()
                actions.append(f"Purged root binary artifact {leftover.name}")
    return actions


def _readme_metrics(model_dir: Path) -> tuple[int, dict[str, Any]]:
    """Epoch count and last row of metrics.csv (0 and {} when missing/unreadable)."""
    metrics_csv = model_dir / "metrics.csv"
    epochs_trained = 0
    metrics: dict[str, Any] = {}
    if metrics_csv.exists():
        try:
            with open(metrics_csv, "r", encoding="utf-8") as cf:
                rows = list(csv.DictReader(cf))
                if rows:
                    epochs_trained = len(rows)
                    metrics = rows[-1]
        except (OSError, csv.Error, UnicodeDecodeError) as csv_err:
            logger.warning(
                "Could not parse %s for README metrics; using empty metrics: %s",
                metrics_csv,
                csv_err,
            )
    return epochs_trained, metrics


class ModelTopologyRestructuringService:
    """Orchestrates structural migration and hygiene across LemGendaryModels model directories."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.project_root = get_project_root()
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = (self.project_root.parent / "LemGendaryModels").resolve()

        self.registry = get_model_registry()
        self.metadata_service = KaggleMetadataService()
        self.doc_service = ModelDocumentationService()
        self.staging_service = VariationStagingService()

    def restructure_model(self, model_key: str) -> dict[str, Any]:
        """Restructure a single model directory into canonical topology."""
        model_dir = self.base_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = model_dir / "checkpoints"
        notebooks_dir = model_dir / "notebooks"
        visualizations_dir = model_dir / "visualizations"
        dist_dir = model_dir / "dist" / "kaggle"

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        visualizations_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)

        actions: list[str] = []

        # 1. Relocate notebooks to notebooks/
        actions.extend(_relocate_notebooks(model_dir, notebooks_dir))

        # 2. Relocate visualization images to visualizations/
        actions.extend(_relocate_visuals(model_dir, visualizations_dir))

        # 3. Relocate and canonicalize training checkpoints strictly to checkpoints/
        ckpt_actions = canonicalize_checkpoints(model_dir, checkpoints_dir)
        actions.extend(ckpt_actions)

        # 4. Stage isolated Kaggle variations
        staging_res = self.staging_service.stage_variations(
            model_key=model_key,
            base_dir=model_dir,
            validate=True,
        )
        if staging_res.staged_files:
            actions.append(f"Staged {len(staging_res.staged_files)} variations into dist/kaggle/")

        actions.extend(self._relocate_runs(model_dir, model_key))
        actions.extend(_remove_rogue_training_dir(model_dir))
        actions.extend(_purge_root_binaries(model_dir))

        # 5. Generate Kaggle model_metadata.yaml
        meta_file = self.metadata_service.write_metadata(model_key=model_key, output_dir=model_dir)
        actions.append(f"Generated {meta_file.name}")

        # 6. Parse metrics and generate canonical README.md
        epochs_trained, metrics = _readme_metrics(model_dir)
        readme_file = self.doc_service.write_readme(
            model_key=model_key,
            output_dir=model_dir,
            epochs_trained=epochs_trained,
            metrics=metrics,
        )
        actions.append(f"Generated {readme_file.name}")

        return {
            "model_key": model_key,
            "actions": actions,
            "isolation_valid": staging_res.success,
            "isolation_report": staging_res.report,
        }

    def _relocate_runs(self, model_dir: Path, model_key: str) -> list[str]:
        """Relocate runs/ directory out of models hub (to <project>/checkpoints)."""
        actions: list[str] = []
        runs_dir = model_dir / "runs"
        if runs_dir.is_dir():
            target_runs = self.project_root / "checkpoints" / model_key / "runs"
            target_runs.mkdir(parents=True, exist_ok=True)
            for item in runs_dir.iterdir():
                dst_item = target_runs / item.name
                if not dst_item.exists():
                    shutil.move(str(item), str(dst_item))
            if remove_tree(runs_dir, purpose="runs/ relocation"):
                actions.append("Relocated runs/ directory to local scratch")
            else:
                actions.append(
                    "Relocated runs/ contents; could not remove the empty runs/ directory"
                )
        return actions

    def restructure_all(self) -> dict[str, Any]:
        """Restructure all registered model directories in LemGendaryModels."""
        all_models = self.registry.get_all_models()
        results: dict[str, Any] = {}

        for key in all_models:
            if key.startswith("_"):
                continue
            logger.info("Restructuring model manifold: %s", key)
            res = self.restructure_model(key)
            results[key] = res

        generate_hub_readme(str(get_project_root()))
        logger.info("Completed topology restructuring across %d models.", len(results))
        return results


def run_restructuring_sweep() -> dict[str, Any]:
    """Convenience accessor to run full topology sweep."""
    service = ModelTopologyRestructuringService()
    return service.restructure_all()
