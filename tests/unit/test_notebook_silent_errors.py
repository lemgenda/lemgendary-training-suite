"""Generated notebook cells never swallow an error silently.

Every ``except`` in the Kaggle and Colab training cells is typed and reports what failed
(``ROBOTS.md``: no ``except Exception: pass``). The background Drive sync of the Colab
notebook used to hide every failure behind ``except Exception as e: pass``.
"""

from __future__ import annotations

import ast
from typing import Any

import pytest

from training.notebooks.builders.base import training_preamble_cells
from training.notebooks.builders.kaggle import build_training_notebook
from training.notebooks.cells.data import build_symlink_cell
from training.notebooks.cells.deps import build_install_cell
from training.notebooks.cells.model import build_checkpoint_recovery_cell
from training.notebooks.cells.sync import (
    build_continuous_sync_cell,
    build_push_cell,
    build_training_cell,
)
from training.notebooks.registry import load_registry, resolve_model_metadata

_MODELS = ("nima_aesthetic_mobile", "forex_predictor", "retinaface")


def _colab_cells(meta: Any) -> list[dict[str, Any]]:
    return [
        *training_preamble_cells(meta, "colab", "Colab"),
        build_install_cell("colab"),
        build_symlink_cell(meta, "colab"),
        build_checkpoint_recovery_cell(meta, "colab"),
        build_continuous_sync_cell(meta),
        build_training_cell(meta, "colab"),
        build_push_cell(meta, "colab"),
    ]


def _code_sources(cells: list[dict[str, Any]]) -> list[str]:
    return ["".join(c["source"]) for c in cells if c.get("cell_type") == "code"]


def _silent_handlers(source: str) -> list[str]:
    """``except`` clauses that are untyped or whose body only passes."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        kind = ast.unparse(node.type) if node.type else "<bare>"
        if kind in ("<bare>", "BaseException"):
            found.append(f"line {node.lineno}: untyped except")
        if all(isinstance(stmt, ast.Pass) for stmt in node.body):
            found.append(f"line {node.lineno}: except {kind}: pass")
    return found


@pytest.mark.parametrize("model_key", _MODELS)
@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_no_silent_except_in_training_cells(model_key: str, target: str) -> None:
    """No handler of the generated training notebook is bare or a plain ``pass``."""
    meta = resolve_model_metadata(model_key, load_registry(), {})
    cells = build_training_notebook(meta)["cells"] if target == "kaggle" else _colab_cells(meta)
    problems = {
        idx: found
        for idx, source in enumerate(_code_sources(cells))
        if (found := _silent_handlers(source))
    }
    assert not problems


def test_drive_sync_reports_typed_failures() -> None:
    """The Colab Drive sync catches OSError and prints the failure instead of passing."""
    meta = resolve_model_metadata("retinaface", load_registry(), {})
    source = "".join(build_continuous_sync_cell(meta)["source"])
    assert "except Exception" not in source
    assert "except OSError as sync_err:" in source
    assert "[SYNC] [WARNING] Drive sync pass failed" in source
