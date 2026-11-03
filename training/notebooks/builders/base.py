"""Common notebook builder serialization and cross-directory synchronization.

Handles JSON encoding, validation, atomic file persistence, and automatic distribution
to sibling LemGendaryDatasets manifolds and workspace training directories.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from training.utils.paths import WORKSPACE_ROOT_ENV
from training.version import __version__

from ..cells.base import make_markdown_cell
from ..cells.deps import build_install_cell
from ..cells.env import build_sentinel_cell
from ..cells.inference import build_onnx_fp16_cell, build_onnx_fp32_cell, build_pth_cell
from ..cells.model import build_hub_prep_cell
from ..cells.repo import build_clone_cell
from ..cells.sync import build_secrets_cell
from ..registry import (
    ModelNotebookMeta,
    load_registry,
    model_dataset_list,
    resolve_model_metadata,
)

logger = logging.getLogger("lemtrain.notebooks.builders.base")


def _get_workspace_root() -> Path:
    """Resolve the canonical workspace root containing sibling LemGendary projects.

    ``LEMGENDARY_WORKSPACE_ROOT`` wins when set, as in :func:`training.utils.paths`.
    """
    override = os.environ.get(WORKSPACE_ROOT_ENV)
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "lemgendary-datasets").exists() and (
            parent / "lemgendary-training-suite"
        ).exists():
            return parent
    return _get_project_root().parent


def _get_project_root() -> Path:
    """Resolve project root directory across training suite or datasets repositories."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (
            (parent / "unified_data.yaml").exists()
            or (parent / "unified_models_v2.yaml").exists()
            or ((parent / "config.yaml").exists() and parent.name == "lemgendary-training-suite")
        ):
            return parent
    return here.parent.parent.parent


def write_notebook(content: dict[str, Any], output_path: str | Path) -> str | None:
    """Validate and write a notebook content dictionary to disk as UTF-8 JSON.

    Returns the formatted JSON string if successful, or None on failure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_str = json.dumps(content, indent=4)
        json.loads(json_str)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        return json_str
    except (TypeError, ValueError, OSError) as e:
        # TypeError: non-JSON-serializable cell content; ValueError: circular refs / decode check.
        logger.error("JSON validation/write failed for %s: %s", out_path, e)
        print(f"[ERROR] JSON Validation failed for {out_path.name}: {e}")
        print(
            "[REMEDY] This usually means the generated notebook syntax is invalid. Check "
            "'unified_models.yaml' for trailing commas or malformed strings."
        )
        return None


def sync_manifold_notebooks(
    model_key: str,
    json_str: str,
    filename: str,
    unified_models_registry: dict[str, Any] | None = None,
) -> None:
    """Sync generated notebook JSON to matching LemGendaryDatasets manifold directories."""
    workspace_root = _get_workspace_root()
    datasets_hub_root = workspace_root / "LemGendaryDatasets"

    if not unified_models_registry:
        try:
            unified_models_registry = load_registry()
        except (ImportError, OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.warning(
                "Could not load model registry for manifold sync of %s: %s", model_key, exc
            )
            unified_models_registry = {}

    if not unified_models_registry:
        return

    ds_list = model_dataset_list(unified_models_registry.get(model_key, {}))

    synced_dirs: set[Path] = set()
    for target_folder in _target_candidates(model_key, ds_list):
        if not target_folder:
            continue
        for m_folder in _manifold_folder_names(target_folder):
            ds_dir = datasets_hub_root / m_folder
            if ds_dir.exists() and ds_dir not in synced_dirs:
                synced_dirs.add(ds_dir)
                _write_notebook(ds_dir / filename, json_str)


def _target_candidates(model_key: str, ds_list: list[str]) -> list[str]:
    """Dataset folders a model's notebooks go to (its datasets plus the model key)."""
    if model_key == "professional_multitask_restoration":
        return [
            "LemGendizedMultitaskRestorationPro",
            "professional_multitask_restoration",
        ]
    target_candidates = list(ds_list)
    if model_key not in target_candidates:
        target_candidates.append(model_key)
    return target_candidates


def _manifold_folder_names(target_folder: str) -> list[str]:
    """Manifold directory spellings tried for one dataset folder name."""
    if "_" in target_folder or "-" in target_folder:
        pascal_name = "".join(
            part.capitalize() for part in target_folder.replace("-", "_").split("_")
        )
    else:
        pascal_name = target_folder
    return [
        target_folder,
        f"{target_folder}Large",
        f"LemGendized{pascal_name}",
        f"LemGendized{pascal_name}Large",
        f"LemGendized{target_folder}Large",
        f"LemGendized{target_folder}",
    ]


def _write_notebook(ds_output_path: Path, json_str: str) -> None:
    """Write the notebook JSON into a manifold directory (failures are logged)."""
    try:
        with open(ds_output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[OK] Synchronized Dataset Manifold Notebook: {ds_output_path}")
    except OSError as exc:
        logger.warning("Failed syncing manifold notebook to %s: %s", ds_output_path, exc)


def sync_workspace_training_notebook(
    subfolder_name: str,
    filename: str,
    json_str: str,
    display_title: str,
) -> None:
    """Sync generated notebook JSON to workspace training folder (e.g. kaggle_training)."""
    workspace_root = _get_workspace_root()
    target_dir = workspace_root / subfolder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / filename
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[OK] Synchronized {display_title} Training Notebook: {out_file}")
    except OSError as exc:
        logger.warning(
            "Failed syncing %s training notebook to %s: %s", display_title, out_file, exc
        )


# --- Shared notebook assembly for the platform builders --------------------------------

_NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.12"},
}


def notebook_document(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap cells in an nbformat 4 notebook document."""
    return {
        "metadata": _NOTEBOOK_METADATA,
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": cells,
    }


def notebooks_dir(export_dir: str | Path) -> str:
    """Return the ``notebooks/`` subfolder of a model export directory when it applies."""
    export_path = Path(export_dir)
    if export_path.name != "notebooks":
        nb_sub = export_path / "notebooks"
        if nb_sub.exists() or export_path.parent.name == "LemGendaryModels":
            return str(nb_sub)
    return str(export_path)


def resolve_training_target(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: Any,
    config: dict[str, Any] | None,
) -> tuple[str, str, ModelNotebookMeta]:
    """Resolve (model key, notebook directory, metadata) for a training notebook.

    Supports the legacy call signature ``(pascal_name, model_key, output_path)``.
    """
    if isinstance(unified_models_registry, (str, Path)):
        actual_model_key = str(export_dir)
        actual_export_dir = str(Path(unified_models_registry).parent)
        meta = resolve_model_metadata(actual_model_key, None, None)
    else:
        actual_model_key = model_key
        actual_export_dir = str(export_dir)
        reg = unified_models_registry if isinstance(unified_models_registry, dict) else None
        if not reg:
            reg = load_registry(config=config)
        meta = resolve_model_metadata(actual_model_key, reg, config)
    return actual_model_key, notebooks_dir(actual_export_dir), meta


def training_preamble_cells(
    meta: ModelNotebookMeta, platform: str, edition: str
) -> list[dict[str, Any]]:
    """Return the title and sections 1-5 (sentinel, secrets, clone, install, hub pull)."""
    return [
        make_markdown_cell(
            [
                f"# LemGendary Master Execution: {meta.pascal_name} (v{__version__} {edition})\n",
                (
                    "This unified notebook handles environment synchronization and automated "
                    "cloud training.\n"
                ),
            ]
        ),
        make_markdown_cell(
            [
                "## 1. Hardware Sentinel\n",
                "Ensure the manifold has the required hardware acceleration.\n",
            ]
        ),
        build_sentinel_cell(platform),
        make_markdown_cell(["## 2. Cloud Auth & Secrets\n"]),
        build_secrets_cell(platform),
        make_markdown_cell(["## 3. Repository Synchronization\n"]),
        build_clone_cell(platform),
        make_markdown_cell(["## 4. Environment & Dependency Installation\n"]),
        build_install_cell(platform),
        make_markdown_cell(["## 5. SOTA Hub Synchronization (Pull)\n"]),
        build_hub_prep_cell(meta, platform),
    ]


def publish_training_notebook(
    cells: list[dict[str, Any]],
    model_key: str,
    notebook_dir: str,
    unified_models_registry: Any,
    platform_title: str,
) -> str | None:
    """Write a training notebook and sync it to the manifolds and the workspace folder."""
    platform = platform_title.lower()
    output_filename = f"{model_key}_{platform}_training.ipynb"
    output_path = os.path.join(notebook_dir, output_filename)

    json_str = write_notebook(notebook_document(cells), output_path)
    if json_str is None:
        return None

    print(f"[OK] Generated {platform_title} Training Notebook: {output_path}")

    sync_manifold_notebooks(
        model_key=model_key,
        json_str=json_str,
        filename=output_filename,
        unified_models_registry=unified_models_registry
        if isinstance(unified_models_registry, dict)
        else None,
    )
    sync_workspace_training_notebook(
        subfolder_name=f"{platform}_training",
        filename=output_filename,
        json_str=json_str,
        display_title=platform_title,
    )
    return output_path


def write_usage_notebook(
    model_key: str,
    export_dir: str | Path,
    meta: ModelNotebookMeta,
    platform_title: str,
) -> str | None:
    """Write the standalone usage guide notebook (PTH and ONNX snippets)."""
    cells = [
        make_markdown_cell(
            [
                f"# LemGendary SOTA Usage: {meta.pascal_name}\n",
                "Implementation guide for production-grade model integration.\n",
            ]
        ),
        make_markdown_cell(
            [
                "## 1. PyTorch Standalone (FP32)\n",
                (
                    "Best for local research, further training, or high-fidelity Python "
                    "backends. "
                    "This format includes the full architecture definition.\n"
                ),
            ]
        ),
        build_pth_cell(meta),
        make_markdown_cell(
            [
                "## 2. ONNX Matrix (FP32 + External Weights)\n",
                (
                    "Optimized for desktop deployment where precision is critical. "
                    "Uses a decoupled `.data` file for stability.\n"
                ),
            ]
        ),
        build_onnx_fp32_cell(meta),
        make_markdown_cell(
            [
                "## 3. ONNX Production (FP16 Embedded)\n",
                (
                    "Production-ready standalone matrix. Optimized for WebGPU, mobile, and "
                    "low-latency edge inference.\n"
                ),
            ]
        ),
        build_onnx_fp16_cell(meta),
    ]
    output_path = os.path.join(
        notebooks_dir(export_dir), f"{model_key}_{platform_title.lower()}_usage.ipynb"
    )
    if write_notebook(notebook_document(cells), output_path) is None:
        return None
    print(f"[OK] Generated {platform_title} Usage Notebook: {output_path}")
    return output_path
