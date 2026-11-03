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
from ..registry import resolve_model_metadata
from .base import (
    publish_training_notebook,
    resolve_training_target,
    training_preamble_cells,
    write_usage_notebook,
)


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
    cells = [
        *training_preamble_cells(meta, "kaggle", "Nuclear-Hardened"),
        make_markdown_cell(["## 6. Multi-Path Data Resolution\n"]),
        build_symlink_cell(meta, "kaggle"),
        make_markdown_cell(["## 7. Checkpoint & Metric Recovery\n"]),
        build_checkpoint_recovery_cell(meta, "kaggle"),
        make_markdown_cell(["## 8. Nuclear Training Matrix\n"]),
        build_training_cell(meta, "kaggle"),
        make_markdown_cell(["## 9. SOTA Models Hub & Google Drive Publishing (4 Variations)\n"]),
        build_push_cell(meta, "kaggle"),
    ]
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
