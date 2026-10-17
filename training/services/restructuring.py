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

import logging
import shutil
from pathlib import Path
from typing import Any

from training.cloud.metadata_generator import KaggleMetadataService
from training.doc_generator import ModelDocumentationService
from training.export.staging import VariationStagingService
from training.registry import get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.services.restructuring")


class ModelTopologyRestructuringService:
    """Orchestrates structural migration and hygiene across all model directories in LemGendaryModels."""

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
        for p in model_dir.glob("*.ipynb"):
            dest = notebooks_dir / p.name
            if dest.exists():
                p.unlink()
                actions.append(f"Purged duplicate notebook from root: {p.name}")
            else:
                shutil.move(str(p), str(dest))
                actions.append(f"Moved notebook {p.name} -> notebooks/")

        # 2. Relocate visualization images to visualizations/
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            for p in model_dir.glob(ext):
                dest = visualizations_dir / p.name
                if dest.exists() and dest != p:
                    p.unlink()
                else:
                    shutil.move(str(p), str(dest))
                actions.append(f"Moved visual asset {p.name} -> visualizations/")

        # 3. Relocate training checkpoints to checkpoints/
        for p in model_dir.glob("*.pth"):
            dest = checkpoints_dir / p.name
            if dest.exists() and dest != p:
                p.unlink()
            else:
                shutil.move(str(p), str(dest))
            actions.append(f"Moved checkpoint {p.name} -> checkpoints/")

        # Ensure canonical pairing in checkpoints/
        best_cand = checkpoints_dir / "best.pth"
        model_best = checkpoints_dir / f"{model_key}_best.pth"
        if best_cand.exists() and not model_best.exists():
            shutil.copy2(best_cand, model_best)
            actions.append(f"Mirrored best.pth -> {model_key}_best.pth")
        elif model_best.exists() and not best_cand.exists():
            shutil.copy2(model_best, best_cand)
            actions.append(f"Mirrored {model_key}_best.pth -> best.pth")

        latest_cand = checkpoints_dir / "latest.pth"
        model_latest = checkpoints_dir / f"{model_key}_latest.pth"
        if latest_cand.exists() and not model_latest.exists():
            shutil.copy2(latest_cand, model_latest)
            actions.append(f"Mirrored latest.pth -> {model_key}_latest.pth")
        elif model_latest.exists() and not latest_cand.exists():
            shutil.copy2(model_latest, latest_cand)
            actions.append(f"Mirrored {model_key}_latest.pth -> latest.pth")

        # 4. Stage isolated Kaggle variations
        staging_res = self.staging_service.stage_variations(
            model_key=model_key,
            base_dir=model_dir,
            validate=True,
        )
        if staging_res.staged_files:
            actions.append(f"Staged {len(staging_res.staged_files)} variations into dist/kaggle/")

        # Relocate runs/ directory out of models hub
        runs_dir = model_dir / "runs"
        if runs_dir.is_dir():
            target_runs = self.project_root / "checkpoints" / model_key / "runs"
            target_runs.mkdir(parents=True, exist_ok=True)
            for item in runs_dir.iterdir():
                dst_item = target_runs / item.name
                if not dst_item.exists():
                    shutil.move(str(item), str(dst_item))
            shutil.rmtree(runs_dir, ignore_errors=True)
            actions.append("Relocated runs/ directory to local scratch")

        # Clean up rogue training/ directory if present
        training_dir = model_dir / "training"
        if training_dir.is_dir():
            shutil.rmtree(training_dir, ignore_errors=True)
            actions.append("Removed rogue training/ directory")

        # Ensure no binaries remain in the root directory
        for ext in ("*.pt", "*.onnx", "*.data", "*.pth"):
            for leftover in model_dir.glob(ext):
                if leftover.is_file():
                    leftover.unlink()
                    actions.append(f"Purged root binary artifact {leftover.name}")

        # 5. Generate Kaggle model-metadata.json
        meta_file = self.metadata_service.write_metadata(model_key=model_key, output_dir=model_dir)
        actions.append(f"Generated {meta_file.name}")

        # 6. Parse metrics and generate canonical README.md
        metrics_csv = model_dir / "metrics.csv"
        epochs_trained = 0
        metrics: dict[str, Any] = {}
        if metrics_csv.exists():
            try:
                import csv

                with open(metrics_csv, "r", encoding="utf-8") as cf:
                    rows = list(csv.DictReader(cf))
                    if rows:
                        epochs_trained = len(rows)
                        metrics = rows[-1]
            except Exception:
                pass

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

    def restructure_all(self) -> dict[str, Any]:
        """Restructure all registered model directories in LemGendaryModels."""
        all_models = self.registry.get_all_models()
        results: dict[str, Any] = {}

        for key in all_models.keys():
            if key.startswith("_"):
                continue
            logger.info("Restructuring model manifold: %s", key)
            res = self.restructure_model(key)
            results[key] = res

        from training.hub_readme_generator import generate_hub_readme

        generate_hub_readme(str(get_project_root()))
        logger.info("Completed topology restructuring across %d models.", len(results))
        return results


def run_restructuring_sweep() -> dict[str, Any]:
    """Convenience accessor to run full topology sweep."""
    service = ModelTopologyRestructuringService()
    return service.restructure_all()
