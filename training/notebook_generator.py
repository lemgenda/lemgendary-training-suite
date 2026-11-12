"""LemGendary Model Training Suite — Legacy Notebook Generator Facade.

This module provides 100% backward compatibility for imports targeting
training.notebook_generator, delegating directly to training.notebooks.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from training.notebooks import (
    generate_colab_inference_notebook,
    generate_colab_training_notebook,
    generate_colab_usage_notebook,
    generate_inference_notebook,
    generate_training_notebook,
    generate_usage_notebook,
    load_registry,
)
from training.notebooks.cells.env import (
    _build_env_var_lines,
    _load_runtime_env,
)
from training.utils.paths import get_models_hub_root, get_project_root
from training.version import __version__

logger = logging.getLogger("lemtrain.notebooks.notebook_generator")

__all__ = [
    "_build_env_var_lines",
    "_load_runtime_env",
    "generate_colab_inference_notebook",
    "generate_colab_training_notebook",
    "generate_colab_usage_notebook",
    "generate_inference_notebook",
    "generate_training_notebook",
    "generate_usage_notebook",
    "main",
]


def _load_config(config_path: Path) -> dict[str, Any]:
    """``config.yaml`` as a dict (empty when missing or not a mapping)."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded_cfg = yaml.safe_load(f)
    return loaded_cfg if isinstance(loaded_cfg, dict) else {}


def _generate_model_notebooks(
    m_key: str, export_root: str, registry: dict[str, Any], config: dict[str, Any]
) -> None:
    """Write the Kaggle and Colab inference/usage notebooks for ``m_key``; drop stray copies."""
    m_dir = os.path.join(export_root, m_key, "notebooks")
    os.makedirs(m_dir, exist_ok=True)
    generate_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
    generate_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
    generate_colab_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
    generate_colab_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)

    # Ensure no duplicate notebook binaries remain in model root
    for stray in (Path(export_root) / m_key).glob("*.ipynb"):
        try:
            stray.unlink()
        except OSError as exc:
            logger.warning("Could not remove stray notebook %s: %s", stray, exc)


def main(argv: list[str] | None = None) -> int:
    """Generate training/inference/usage notebooks for one model or the whole registry."""
    parser = argparse.ArgumentParser(
        description=f"LemGendary Notebook Orchestrator (v{__version__})"
    )
    parser.add_argument("--model", type=str, help="Generate notebooks for a specific model key.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key context if invoked.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate the entire Notebook Matrix for all registry models.",
    )
    parser.add_argument("--dir", type=str, help="Override export directory.")
    args = parser.parse_args(argv)

    project_root = get_project_root()
    config = _load_config(project_root / "config.yaml")
    registry = load_registry(config, project_root)

    configured_export = config.get("export_dir")
    default_export = str(
        (project_root / configured_export).resolve()
        if configured_export
        else get_models_hub_root(project_root)
    )
    export_root = args.dir if args.dir else default_export

    models_to_gen: list[str] = []
    if args.all:
        models_to_gen = [k for k in registry if not k.startswith("_")]
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(models_to_gen)} models...")
    elif args.model:
        if args.model in registry:
            models_to_gen = [args.model]
        else:
            print(f"[ERROR] Model '{args.model}' not found in registry.")
            print("[REMEDY] Verify the spelling of the model key in 'unified_models_v2.yaml'.")
            return 1
    else:
        parser.print_help()
        return 0

    for m_key in models_to_gen:
        _generate_model_notebooks(m_key, export_root, registry, config)

    print("\n[SUCCESS] Notebook Matrix Synchronized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
