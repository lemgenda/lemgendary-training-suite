"""Importing the suite does not import the ``kaggle`` package.

``kaggle`` authenticates when it is imported and prints its sign-in help without
credentials, so only the code that calls the Kaggle API loads it. Each module is imported
in a fresh interpreter, whose ``sys.modules`` then must not hold ``kaggle``.
"""

from __future__ import annotations

import pytest

from tests.unit.import_probe import loaded_after_import


@pytest.mark.parametrize(
    "module",
    [
        "training.model_card",
        "training.export",
        "training.cli.lemtrain",
        "training.cloud",
        "training.cloud.kaggle_api_sync",
    ],
)
def test_import_leaves_kaggle_unloaded(module: str) -> None:
    """The module imports without ``kaggle`` and prints nothing of it."""
    loaded, output = loaded_after_import(module, ("kaggle",))
    assert loaded == []
    assert "Authentication required" not in output
