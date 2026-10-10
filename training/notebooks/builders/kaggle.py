"""Kaggle platform notebook builder.

Generates automated cloud training notebooks and standalone inference usage notebooks
specifically tailored for Kaggle environments.
"""

import os
from pathlib import Path
from typing import Any

from ..cells.base import make_markdown_cell
from ..cells.data import build_symlink_cell
from ..cells.deps import build_install_cell
from ..cells.env import build_sentinel_cell
from ..cells.inference import (
    build_onnx_fp16_cell,
    build_onnx_fp32_cell,
    build_pth_cell,
)
from ..cells.model import (
    build_checkpoint_recovery_cell,
    build_hub_prep_cell,
)
from ..cells.repo import build_clone_cell
from ..cells.sync import (
    build_secrets_cell,
    build_training_cell,
)
from ..registry import resolve_model_metadata
from .base import (
    sync_manifold_notebooks,
    sync_workspace_training_notebook,
    write_notebook,
)


def generate_inference_notebook(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: Any = None,
    config: dict[str, Any] | None = None,
    *extra_args: Any,
    **extra_kwargs: Any,
) -> str | None:
    """Generate a nuclear-hardened training execution notebook for Kaggle."""
    if extra_args or (isinstance(unified_models_registry, (str, Path)) and not isinstance(unified_models_registry, dict)):
        # Legacy call signature: (pascal_name, model_key, output_path)
        actual_model_key = str(export_dir)
        target_path = Path(unified_models_registry if not extra_args else extra_args[0])
        actual_export_dir = str(target_path.parent)
        meta = resolve_model_metadata(actual_model_key, None, None)
    else:
        actual_model_key = model_key
        actual_export_dir = str(export_dir)
        meta = resolve_model_metadata(actual_model_key, unified_models_registry, config)

    notebook_content = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            make_markdown_cell([
                f"# LemGendary Master Execution: {meta.pascal_name} (v16.2.9 Nuclear-Hardened)\n",
                "This unified notebook handles environment synchronization and automated cloud training.\n",
            ]),
            make_markdown_cell([
                "## 1. Hardware Sentinel\n",
                "Ensure the manifold has the required hardware acceleration.\n",
            ]),
            build_sentinel_cell("kaggle"),
            make_markdown_cell(["## 2. Cloud Auth & Secrets\n"]),
            build_secrets_cell("kaggle"),
            make_markdown_cell(["## 3. Environment Synchronization\n"]),
            build_clone_cell("kaggle"),
            build_install_cell("kaggle"),
            make_markdown_cell(["## 4. SOTA Hub Synchronization (Pull)\n"]),
            build_hub_prep_cell(meta, "kaggle"),
            make_markdown_cell(["## 5. Multi-Path Data Resolution\n"]),
            build_symlink_cell(meta, "kaggle"),
            make_markdown_cell(["## 6. Checkpoint & Metric Recovery\n"]),
            build_checkpoint_recovery_cell(meta, "kaggle"),
            make_markdown_cell(["## 7. Nuclear Training Matrix\n"]),
            build_training_cell(meta, "kaggle"),
        ],
    }

    output_filename = f"{actual_model_key}_kaggle_training.ipynb"
    output_path = os.path.join(actual_export_dir, output_filename)

    json_str = write_notebook(notebook_content, output_path)
    if json_str is None:
        return None

    print(f"[OK] Generated Kaggle Training Notebook: {output_path}")

    sync_manifold_notebooks(
        model_key=actual_model_key,
        json_str=json_str,
        filename=output_filename,
        unified_models_registry=unified_models_registry if isinstance(unified_models_registry, dict) else None,
    )

    sync_workspace_training_notebook(
        subfolder_name="kaggle_training",
        filename=output_filename,
        json_str=json_str,
        display_title="Kaggle",
    )

    return output_path


def generate_usage_notebook(
    model_key: str,
    export_dir: str,
    unified_models_registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a standalone model usage guide notebook with PTH and ONNX snippets for Kaggle."""
    meta = resolve_model_metadata(model_key, unified_models_registry, config)

    notebook_content = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            make_markdown_cell([
                f"# LemGendary SOTA Usage: {meta.pascal_name}\n",
                "Implementation guide for production-grade model integration.\n",
            ]),
            make_markdown_cell([
                "## 1. PyTorch Standalone (FP32)\n",
                "Best for local research, further training, or high-fidelity Python backends. This format includes the full architecture definition.\n",
            ]),
            build_pth_cell(meta),
            make_markdown_cell([
                "## 2. ONNX Matrix (FP32 + External Weights)\n",
                "Optimized for desktop deployment where precision is critical. Uses a decoupled `.data` file for stability.\n",
            ]),
            build_onnx_fp32_cell(meta),
            make_markdown_cell([
                "## 3. ONNX Production (FP16 Embedded)\n",
                "Production-ready standalone matrix. Optimized for WebGPU, mobile, and low-latency edge inference.\n",
            ]),
            build_onnx_fp16_cell(meta),
        ],
    }

    output_path = os.path.join(export_dir, f"{model_key}-kaggle-usage.ipynb")
    json_str = write_notebook(notebook_content, output_path)
    if json_str is None:
        return None

    print(f"[OK] Generated Kaggle Usage Notebook: {output_path}")
    return output_path


generate_training_notebook = generate_inference_notebook
generate_kaggle_usage_notebook = generate_usage_notebook
