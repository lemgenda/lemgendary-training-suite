"""Every generated usage snippet imports names that exist (R4-FU item 5).

The model-card snippets (one per task family) and the Kaggle variation snippets are parsed;
each ``import`` must name an importable module and each ``from ... import`` a name the module
has. The restoration snippet imported ``models.factory.create_model`` (the factory function
is ``get_model``) and two snippets used ``np`` without importing NumPy.
"""

from __future__ import annotations

import ast
import importlib

import pytest

from training.cloud.metadata_generator import KaggleMetadataService
from training.cloud.variations import (
    ONNX_FP16,
    ONNX_FP32,
    PYTORCH_DEFAULT,
    PYTORCH_FP32,
    KaggleVariation,
)
from training.model_card_snippets import SnippetInput, usage_snippet
from training.registry.preprocessing import IMAGENET

_SNIPPET_TASKS = {
    "quality": "nima_technical",
    "restoration": "ultrazoom",
    "enhancement": "mirnet_exposure",
    "forex": "forex_predictor",
    "classification": "universal_nsfw_classification",
}


def _code(snippet: str) -> str:
    """The Python source inside a fenced ```python block."""
    body = snippet.split("```python", 1)[1]
    return body.split("```", 1)[0]


def _imports(source: str) -> list[tuple[str, str | None]]:
    """``(module, name)`` of every import (``name`` None for ``import module``)."""
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found += [(node.module, alias.name) for alias in node.names]
    return found


def _assert_imports_resolve(source: str) -> None:
    imports = _imports(source)
    assert imports, "a usage snippet imports something"
    for module_name, name in imports:
        module = importlib.import_module(module_name)
        assert name is None or hasattr(module, name), f"{module_name}.{name} does not exist"
    if "np." in source:
        assert ("numpy", None) in imports, "the snippet uses np without importing NumPy"


@pytest.mark.parametrize(("task", "model_key"), sorted(_SNIPPET_TASKS.items()))
def test_model_card_snippet_imports_resolve(task: str, model_key: str) -> None:
    """Model-card snippets import only existing modules and names."""
    snippet = usage_snippet(task, model_key, SnippetInput(224, 224, preprocessing=IMAGENET))
    _assert_imports_resolve(_code(snippet))


@pytest.mark.parametrize("variation", [PYTORCH_DEFAULT, PYTORCH_FP32, ONNX_FP16, ONNX_FP32])
def test_kaggle_variation_snippet_imports_resolve(variation: KaggleVariation) -> None:
    """The Kaggle model README snippets import only existing modules and names."""
    snippet = KaggleMetadataService.variation_usage("owner", "slug", "Tiny", variation)
    _assert_imports_resolve(_code(snippet))


def test_restoration_snippet_uses_the_factory_function() -> None:
    """The factory function is ``get_model``; ``create_model`` does not exist."""
    snippet = usage_snippet("restoration", "ultrazoom", SnippetInput(256, 256))
    assert "from models.factory import get_model" in snippet
    assert "create_model" not in snippet
