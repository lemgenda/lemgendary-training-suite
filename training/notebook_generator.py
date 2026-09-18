"""LemGendary Model Training Suite — Legacy Notebook Generator Facade.

This module provides 100% backward compatibility for imports targeting
training.notebook_generator, delegating directly to training.notebooks.
"""

import argparse
import os
import sys
from typing import Any

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
from training.utils.paths import get_project_root

__all__ = [
    "_load_runtime_env",
    "_build_env_var_lines",
    "generate_inference_notebook",
    "generate_usage_notebook",
    "generate_training_notebook",
    "generate_colab_inference_notebook",
    "generate_colab_usage_notebook",
    "generate_colab_training_notebook",
]

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="LemGendary Notebook Orchestrator (v16.2.9 Nuclear)")
    parser.add_argument("--model", type=str, help="Generate notebooks for a specific model key.")
    parser.add_argument("--all", action="store_true", help="Regenerate the entire Notebook Matrix for all registry models.")
    parser.add_argument("--dir", type=str, help="Override export directory.")
    args = parser.parse_args()

    project_root = get_project_root()
    config_path = project_root / "config.yaml"

    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = yaml.safe_load(f)
            if isinstance(loaded_cfg, dict):
                config = loaded_cfg

    registry = load_registry(config, project_root)

    default_export = str((project_root / config.get("export_dir", "../LemGendaryModels")).resolve())
    export_root = args.dir if args.dir else default_export

    models_to_gen: list[str] = []
    if args.all:
        models_to_gen = [k for k in registry.keys() if k != "_registry_metadata"]
        print(f"[NUCLEAR] Initiating Global Notebook Refresh for {len(models_to_gen)} models...")
    elif args.model:
        if args.model in registry:
            models_to_gen = [args.model]
        else:
            print(f"[ERROR] Model '{args.model}' not found in registry.")
            print("[REMEDY] Verify the spelling of the model key in 'unified_models.yaml'.")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    for m_key in models_to_gen:
        m_dir = os.path.join(export_root, m_key)
        os.makedirs(m_dir, exist_ok=True)
        generate_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_colab_inference_notebook(m_key, m_dir, unified_models_registry=registry, config=config)
        generate_colab_usage_notebook(m_key, m_dir, unified_models_registry=registry, config=config)

    print("\n[SUCCESS] Notebook Matrix Synchronized.")