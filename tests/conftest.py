"""Shared pytest fixtures: every test runs against a fake, empty workspace.

``LEMGENDARY_WORKSPACE_ROOT`` points at a per-test temporary folder, so nothing a test does can
read or write the real sibling checkouts (``LemGendaryModels``, ``lemgendary-docs``, ...).
``LEMGENDARY_ECOSYSTEM_FILE`` points at ``tests/fixtures/ecosystem.json``, so ports, the Kaggle
owner and the docs URL come from the fixture (deliberately different from the real values, which
proves the code reads the file instead of literals).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from training import ecosystem
from training.ecosystem import ECOSYSTEM_FILE_ENV
from training.utils.paths import WORKSPACE_ROOT_ENV

ECOSYSTEM_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ecosystem.json"


@pytest.fixture(autouse=True)
def fake_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the test in a temporary workspace with the fixture ecosystem file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(workspace))
    monkeypatch.setenv(ECOSYSTEM_FILE_ENV, str(ECOSYSTEM_FIXTURE))
    ecosystem.clear_cache()
    yield workspace
    ecosystem.clear_cache()
