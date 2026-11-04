"""Google Colab platform notebook builder.

Generates automated cloud training notebooks with Google Drive continuous synchronization,
and standalone inference usage notebooks specifically tailored for Google Colab environments.
"""

from pathlib import Path
from typing import Any

from ..cells.base import make_markdown_cell
from ..cells.data import build_symlink_cell
from ..cells.model import build_checkpoint_recovery_cell
from ..cells.sync import (
    build_continuous_sync_cell,
    build_fuse_mount_cell,
    build_push_cell,
    build_training_cell,
)
from ..registry import resolve_model_metadata
from .base import (
    publish_training_notebook,
    resolve_training_target,
    training_preamble_cells,
    write_usage_notebook,
)


def generate_colab_inference_notebook(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: Any = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a nuclear-hardened training execution notebook for Google Colab."""
    actual_model_key, notebook_dir, meta = resolve_training_target(
        model_key, export_dir, unified_models_registry, config
    )
    cells = [
        *training_preamble_cells(meta, "colab", "Nuclear-Hardened Colab Edition"),
        make_markdown_cell(
            [
                "## 6. Google Drive Mount\n",
                "Mount Google Drive FUSE for streaming datasets directly.\n",
            ]
        ),
        build_fuse_mount_cell(),
        make_markdown_cell(["## 7. Kaggle Dataset Acquisition & Manifold Resolution\n"]),
        build_symlink_cell(meta, "colab"),
        make_markdown_cell(["## 8. Checkpoint & Metric Recovery\n"]),
        build_checkpoint_recovery_cell(meta, "colab"),
        make_markdown_cell(["## 9. Continuous Drive Synchronization\n"]),
        build_continuous_sync_cell(meta),
        make_markdown_cell(["## 10. Nuclear Training Matrix\n"]),
        build_training_cell(meta, "colab"),
        make_markdown_cell(["## 11. SOTA Models Hub & Google Drive Publishing (4 Variations)\n"]),
        build_push_cell(meta, "colab"),
    ]
    return publish_training_notebook(
        cells, actual_model_key, notebook_dir, unified_models_registry, "Colab"
    )


def generate_colab_usage_notebook(
    model_key: str,
    export_dir: str | Path,
    unified_models_registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Generate a standalone model usage guide notebook with PTH and ONNX snippets for Colab."""
    meta = resolve_model_metadata(model_key, unified_models_registry, config)
    return write_usage_notebook(model_key, export_dir, meta, "Colab")


generate_colab_training_notebook = generate_colab_inference_notebook
