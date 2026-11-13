"""Kaggle platform notebook builder.

Generates automated cloud training notebooks and standalone inference usage notebooks
specifically tailored for Kaggle environments.
"""

from pathlib import Path
from typing import Any

from ..cells.base import make_markdown_cell
from ..cells.data import build_symlink_cell
from ..cells.model import build_checkpoint_recovery_cell
from ..cells.sync import build_push_cell, build_training_cell
from ..launch_options import DEFAULT_LAUNCH, LaunchOptions
from ..registry import ModelNotebookMeta, resolve_model_metadata
from .base import (
    notebook_document,
    publish_training_notebook,
    resolve_training_target,
    training_preamble_cells,
    write_usage_notebook,
)


def training_cells(
    meta: ModelNotebookMeta, options: LaunchOptions = DEFAULT_LAUNCH
) -> list[dict[str, Any]]:
    """The cells of the Kaggle training notebook of ``meta`` (no file is written).

    ``options`` are the launch options: clone refs, trainer switches and session window
    (defaults: ``main``, no switches, ``config.yaml``'s ``kaggle_session``).
    """
    return [
        *training_preamble_cells(meta, "kaggle", "Nuclear-Hardened", options.git_refs),
        make_markdown_cell(["## 6. Multi-Path Data Resolution\n"]),
        build_symlink_cell(meta, "kaggle"),
        make_markdown_cell(["## 7. Checkpoint & Metric Recovery\n"]),
        build_checkpoint_recovery_cell(meta, "kaggle"),
        make_markdown_cell(["## 8. Nuclear Training Matrix\n"]),
        build_training_cell(meta, "kaggle", options),
        make_markdown_cell(["## 9. SOTA Models Hub & Google Drive Publishing (4 Variations)\n"]),
        build_push_cell(meta, "kaggle"),
    ]


def build_training_notebook(
    meta: ModelNotebookMeta, options: LaunchOptions = DEFAULT_LAUNCH
) -> dict[str, Any]:
    """The Kaggle training notebook document of ``meta``.

    The one generator of the Kaggle training notebook: the notebook files written by
    :func:`generate_inference_notebook` and the kernel bundles of
    :mod:`training.cloud.kernel_bundle` (GUI, CLI) are built from it, so a kernel launched
    from the GUI runs exactly the notebook a user would upload.
    """
    return notebook_document(training_cells(meta, options))


def generate_inference_notebook(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: Any = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a nuclear-hardened training execution notebook for Kaggle."""
    actual_model_key, notebook_dir, meta = resolve_training_target(
        model_key, export_dir, unified_models_registry, config
    )
    cells = training_cells(meta)
    return publish_training_notebook(
        cells, actual_model_key, notebook_dir, unified_models_registry, "Kaggle"
    )


def generate_usage_notebook(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a standalone model usage guide notebook with PTH and ONNX snippets for Kaggle."""
    meta = resolve_model_metadata(model_key, unified_models_registry, config)
    return write_usage_notebook(model_key, export_dir, meta, "Kaggle")


generate_training_notebook = generate_inference_notebook
generate_kaggle_usage_notebook = generate_usage_notebook
