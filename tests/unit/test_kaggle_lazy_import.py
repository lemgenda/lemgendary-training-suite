"""Importing the suite does not import the ``kaggle`` package.

``kaggle`` authenticates when it is imported and prints its sign-in help without
credentials, so only the code that calls the Kaggle API loads it. Each module is imported
in a fresh interpreter, whose ``sys.modules`` then must not hold ``kaggle``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "print(sorted(m for m in sys.modules if m == 'kaggle' or m.startswith('kaggle.')))\n"
)


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
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE, module],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert probe.stdout.strip().splitlines()[-1] == "[]"
    assert "Authentication required" not in probe.stdout + probe.stderr
