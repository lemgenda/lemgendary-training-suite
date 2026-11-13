"""Generated training notebook cells of both platforms, for tests that audit cell sources."""

from __future__ import annotations

from typing import Any

from training.notebooks.builders.base import training_preamble_cells
from training.notebooks.builders.kaggle import build_training_notebook
from training.notebooks.cells.data import build_symlink_cell
from training.notebooks.cells.model import build_checkpoint_recovery_cell
from training.notebooks.cells.sync import (
    build_continuous_sync_cell,
    build_push_cell,
    build_training_cell,
)


def colab_training_cells(meta: Any) -> list[dict[str, Any]]:
    """The cells of the Colab training notebook of ``meta`` (no file is written)."""
    return [
        *training_preamble_cells(meta, "colab", "Colab"),
        build_symlink_cell(meta, "colab"),
        build_checkpoint_recovery_cell(meta, "colab"),
        build_continuous_sync_cell(meta),
        build_training_cell(meta, "colab"),
        build_push_cell(meta, "colab"),
    ]


def training_code_sources(meta: Any, target: str) -> list[str]:
    """Source text of every code cell of the ``target`` training notebook of ``meta``."""
    if target == "kaggle":
        cells = build_training_notebook(meta)["cells"]
    else:
        cells = colab_training_cells(meta)
    return ["".join(c["source"]) for c in cells if c.get("cell_type") == "code"]
