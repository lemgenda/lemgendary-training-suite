"""Google Colab platform notebook builder.

Generates automated cloud training notebooks with Google Drive continuous synchronization,
and standalone inference usage notebooks specifically tailored for Google Colab environments.
"""

import os
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
    build_continuous_sync_cell,
    build_fuse_mount_cell,
    build_secrets_cell,
    build_training_cell,
)
from ..registry import resolve_model_metadata
from .base import (
    sync_manifold_notebooks,
    sync_workspace_training_notebook,
    write_notebook,
)


def generate_colab_inference_notebook(
    model_key: str,
    export_dir: str,
    unified_models_registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a nuclear-hardened training execution notebook for Google Colab."""
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
                f"# LemGendary Master Execution: {meta.pascal_name} (v16.2.9 Nuclear-Hardened Colab Edition)\n",
                "This unified notebook handles environment synchronization and automated cloud training.\n",
            ]),
            make_markdown_cell([
                "## 1. Hardware Sentinel\n",
                "Ensure the manifold has the required hardware acceleration.\n",
            ]),
            build_sentinel_cell("colab"),
            make_markdown_cell(["## 2. Cloud Auth & Secrets\n"]),
            build_secrets_cell("colab"),
            make_markdown_cell(["## 3. Environment Synchronization\n"]),
            build_clone_cell("colab"),
            build_install_cell("colab"),
            make_markdown_cell(["## 4. SOTA Hub Synchronization (Pull)\n"]),
            build_hub_prep_cell(meta, "colab"),
            make_markdown_cell([
                "## 4.5 Google Drive Mount\n",
                "Mount Google Drive FUSE for streaming datasets directly.\n",
            ]),
            build_fuse_mount_cell(),
            make_markdown_cell(["## 5. Kaggle Dataset Acquisition & Manifold Resolution\n"]),
            build_symlink_cell(meta, "colab"),
            make_markdown_cell(["## 6. Checkpoint & Metric Recovery\n"]),
            build_checkpoint_recovery_cell(meta, "colab"),
            make_markdown_cell(["## 7. Continuous Drive Synchronization\n"]),
            build_continuous_sync_cell(meta),
            make_markdown_cell(["## 8. Nuclear Training Matrix\n"]),
            build_training_cell(meta, "colab"),
        ],
    }

    output_filename = f"{model_key}_colab_training.ipynb"
    output_path = os.path.join(export_dir, output_filename)

    json_str = write_notebook(notebook_content, output_path)
    if json_str is None:
        return None

    print(f"[OK] Generated Colab Training Notebook: {output_path}")

    sync_manifold_notebooks(
        model_key=model_key,
        json_str=json_str,
        filename=output_filename,
        unified_models_registry=unified_models_registry,
    )

    sync_workspace_training_notebook(
        subfolder_name="colab_training",
        filename=output_filename,
        json_str=json_str,
        display_title="Colab",
    )

    return output_path


def generate_colab_usage_notebook(
    model_key: str,
    export_dir: str,
    unified_models_registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a standalone model usage guide notebook with PTH and ONNX snippets for Colab."""
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

    output_path = os.path.join(export_dir, f"{model_key}-colab-usage.ipynb")
    json_str = write_notebook(notebook_content, output_path)
    if json_str is None:
        return None

    print(f"[OK] Generated Colab Usage Notebook: {output_path}")
    return output_path


generate_colab_training_notebook = generate_colab_inference_notebook
