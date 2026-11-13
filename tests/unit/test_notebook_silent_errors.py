"""Generated notebook cells never swallow an error silently.

Every ``except`` in the Kaggle and Colab training cells is typed and reports what failed
(``ROBOTS.md``: no ``except Exception: pass``). The background Drive sync of the Colab
notebook used to hide every failure behind ``except Exception as e: pass``.
"""

from __future__ import annotations

import ast

import pytest

from tests.fixtures.notebook_cells import training_code_sources
from training.notebooks.cells.sync import build_continuous_sync_cell
from training.notebooks.registry import load_registry, resolve_model_metadata

_MODELS = ("nima_aesthetic_mobile", "forex_predictor", "retinaface")


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
    problems = {
        idx: found
        for idx, source in enumerate(training_code_sources(meta, target))
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
