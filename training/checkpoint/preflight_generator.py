"""Universal Pre-Flight Asset Generator for LemGendary AI Training Suite.

Automates the pre-training generation of deployment documentation (README.md)
and reproducible cloud execution notebooks ([ModelName]_kaggle_training.ipynb and
[ModelName]_colab_training.ipynb) before Epoch 1 begins.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

from training.doc_generator import build_model_readme
from training.notebooks import (
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
    generate_training_notebook,
    generate_usage_notebook,
)
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.checkpoint.preflight")


def _load_registry_and_config(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load unified models registry and suite configuration."""
    cfg_path = project_root / "config.yaml"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Could not parse config.yaml: %s", exc)

    rel_reg = cfg.get("unified_models", "unified_models_v2.yaml")
    reg_path = project_root / rel_reg
    reg_data: dict[str, Any] = {}
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as rf:
                reg_data = yaml.safe_load(rf) or {}
        except Exception as exc:
            logger.warning("Could not parse %s: %s", rel_reg, exc)

    return reg_data, cfg


def generate_preflight_assets(
    model_key: str,
    project_root: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Generate pre-flight README and reproducible cloud execution notebooks for a model.

    Artifacts generated strictly under LemGendaryModels/[model_key]/:
    - README.md: Model documentation with topology and metrics.
    - [model_key]_kaggle_training.ipynb & [model_key]_training.ipynb: Kaggle execution notebook.
    - [model_key]_colab_training.ipynb: Google Colab execution notebook.
    - [model_key]-usage.ipynb: Standalone inference and usage notebook.
    - [model_key]-colab-usage.ipynb: Standalone Colab usage notebook.

    Args:
        model_key: Canonical model key registered in unified_models_v2.yaml.
        project_root: Root path of lemgendary-training-suite.
        force: If True, overwrite existing files.

    Returns:
        dict[str, Path]: Dictionary mapping artifact identifier to generated Path.
    """
    root = (project_root or get_project_root()).resolve()
    hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (hub_dir / "training").mkdir(parents=True, exist_ok=True)

    registry, config = _load_registry_and_config(root)
    generated_assets: dict[str, Path] = {}

    # 1. Model README.md
    readme_path = hub_dir / "README.md"
    if force or not readme_path.exists():
        epochs_trained = 0
        metrics: dict[str, Any] = {}
        metrics_csv = hub_dir / "metrics.csv"
        if metrics_csv.exists():
            try:
                with open(metrics_csv, "r", encoding="utf-8") as cf:
                    reader = list(csv.DictReader(cf))
                    if reader:
                        last_ep = reader[-1].get("Epoch")
                        if last_ep and str(last_ep).isdigit():
                            epochs_trained = int(last_ep) + 1
                        else:
                            epochs_trained = len(reader)
                        metrics = reader[-1]
            except Exception as exc:
                logger.debug("Metrics parsing notice for %s: %s", model_key, exc)

        content = build_model_readme(
            model_key=model_key,
            unified_models=registry,
            epochs_trained=epochs_trained,
            metrics=metrics,
        )
        readme_path.write_text(content, encoding="utf-8")
        logger.info("Generated pre-flight README: %s", readme_path)
    generated_assets["readme"] = readme_path

    # 2. Kaggle Training Notebooks
    kaggle_train_path = hub_dir / f"{model_key}_training.ipynb"
    kaggle_alt_train_path = hub_dir / f"{model_key}_kaggle_training.ipynb"
    if force or not kaggle_train_path.exists() or not kaggle_alt_train_path.exists():
        try:
            out_nb = generate_training_notebook(
                model_key=model_key,
                export_dir=str(hub_dir),
                unified_models_registry=registry,
                config=config,
            )
            if out_nb and Path(out_nb).exists():
                shutil.copy2(out_nb, kaggle_alt_train_path)
                generated_assets["kaggle_training"] = kaggle_alt_train_path
                generated_assets["kaggle_training_std"] = Path(out_nb)
        except Exception as exc:
            logger.warning("Kaggle training notebook generation notice for %s: %s", model_key, exc)

    # 3. Google Colab Training Notebook
    colab_train_path = hub_dir / f"{model_key}_colab_training.ipynb"
    if force or not colab_train_path.exists():
        try:
            out_colab = generate_colab_training_notebook(
                model_key=model_key,
                export_dir=str(hub_dir),
                unified_models_registry=registry,
                config=config,
            )
            if out_colab and Path(out_colab).exists():
                generated_assets["colab_training"] = Path(out_colab)
        except Exception as exc:
            logger.warning("Colab training notebook generation notice for %s: %s", model_key, exc)

    # 4. Usage Notebooks (Kaggle & Colab Standalone Inference)
    usage_nb = hub_dir / f"{model_key}-usage.ipynb"
    if force or not usage_nb.exists():
        try:
            generate_usage_notebook(
                model_key=model_key,
                export_dir=str(hub_dir),
                unified_models_registry=registry,
                config=config,
            )
            if usage_nb.exists():
                generated_assets["usage"] = usage_nb
        except Exception as exc:
            logger.debug("Usage notebook generation notice for %s: %s", model_key, exc)

    colab_usage_nb = hub_dir / f"{model_key}-colab-usage.ipynb"
    if force or not colab_usage_nb.exists():
        try:
            generate_colab_usage_notebook(
                model_key=model_key,
                export_dir=str(hub_dir),
                unified_models_registry=registry,
                config=config,
            )
            if colab_usage_nb.exists():
                generated_assets["colab_usage"] = colab_usage_nb
        except Exception as exc:
            logger.debug("Colab usage notebook generation notice for %s: %s", model_key, exc)

    return generated_assets


def generate_all_preflight_assets(
    project_root: Path | None = None,
    force: bool = False,
) -> dict[str, dict[str, Path]]:
    """Generate pre-flight assets across all models registered in unified_models_v2.yaml."""
    root = (project_root or get_project_root()).resolve()
    registry, _ = _load_registry_and_config(root)

    results: dict[str, dict[str, Path]] = {}
    models_dict = registry.get("models")
    if not isinstance(models_dict, dict):
        models_dict = registry

    model_keys = sorted([k for k in models_dict.keys() if not k.startswith("_")])
    for model_key in model_keys:
        try:
            res = generate_preflight_assets(model_key, root, force=force)
            results[model_key] = res
        except Exception as exc:
            logger.error("Failed preflight asset generation for %s: %s", model_key, exc)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LemGendary Pre-Flight Asset Generator")
    parser.add_argument("--model", type=str, help="Model key to generate preflight assets for")
    parser.add_argument("--all", action="store_true", help="Generate preflight assets for all registered models")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing assets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_root = get_project_root()

    if args.all:
        print("[PREFLIGHT] Synchronizing pre-flight assets for all models...", flush=True)
        summary = generate_all_preflight_assets(base_root, force=args.force)
        print(f"[PREFLIGHT] Complete. Synchronized {len(summary)} model asset bundles.", flush=True)
    elif args.model:
        print(f"[PREFLIGHT] Synchronizing pre-flight assets for {args.model}...", flush=True)
        assets = generate_preflight_assets(args.model, base_root, force=args.force)
        print(f"[PREFLIGHT] Complete. Generated {len(assets)} assets for {args.model}:", flush=True)
        for k, p in assets.items():
            print(f"  - {k:20}: {p}", flush=True)
    else:
        parser.print_help()
        sys.exit(0)
