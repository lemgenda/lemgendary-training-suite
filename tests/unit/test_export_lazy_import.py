"""Light export submodules must not drag torch or the model factory into the interpreter.

Regression: ``python -m training.kaggle_cloud_manager --help`` exceeded 30 s because
``training.export.__init__`` eagerly imported ``models.factory`` (torch, torchmetrics).
"""

from __future__ import annotations

import pytest

import training.export as export_pkg
from tests.unit.import_probe import loaded_after_import

_HEAVY = ("torch", "models.factory", "torchmetrics")


@pytest.mark.parametrize(
    "module", ["training.export.validation", "training.cloud.kaggle_model_sync"]
)
def test_light_modules_do_not_import_heavy_stack(module: str) -> None:
    """Importing the module leaves torch, models.factory and torchmetrics unloaded."""
    loaded, _output = loaded_after_import(module, _HEAVY)
    assert loaded == []


def test_lazy_attributes_resolve() -> None:
    """Public names resolve on first access and unknown names raise AttributeError."""
    assert export_pkg.ExportError.__name__ == "ExportError"
    assert callable(export_pkg.export_all)
    assert not hasattr(export_pkg, "does_not_exist")
