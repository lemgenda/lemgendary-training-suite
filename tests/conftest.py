"""Shared pytest fixtures: every test runs against a fake, empty workspace.

``LEMGENDARY_WORKSPACE_ROOT`` points at a per-test temporary folder, so nothing a test does can
read or write the real sibling checkouts (``LemGendaryModels``, ``lemgendary-docs``, ...).
``LEMGENDARY_ECOSYSTEM_FILE`` points at ``tests/fixtures/ecosystem.json``, so ports, the Kaggle
owner and the docs URL come from the fixture (deliberately different from the real values, which
proves the code reads the file instead of literals).
``LEMGENDARY_TASK_TAXONOMY_FILE`` is unset, so the datasets task taxonomy is looked up in the empty
workspace; tests that need it point the variable at ``tests/fixtures/task_taxonomy.json``.
The ``.secrets.yaml`` vault reader (``training.config.secrets.load_secrets_records``, which the
Kaggle account lookup and ``get_secret`` both use) returns no records, so no test can read the
real vault next to the checkout or mount its tokens into ``os.environ`` (tests that need accounts
patch it with fixture records). The GitHub and Google Drive token variables are unset, so
``get_secret`` sees only what a test sets itself.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from training import ecosystem
from training.config import secrets as secrets_vault
from training.data import task_resolution
from training.ecosystem import ECOSYSTEM_FILE_ENV
from training.utils.paths import WORKSPACE_ROOT_ENV

ECOSYSTEM_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ecosystem.json"

TOKEN_SECRET_VARS = ("GITHUB_PAT", "SUITE_PAT", "GOOGLE_DRIVE", "GOOGLE_DRIVE_JSON")
"""Environment variables ``get_secret`` serves for the GitHub and Google Drive credentials."""


def _no_vault_records(_project_root: Path | None = None) -> list[dict[str, object]]:
    """Stand-in for the vault reader: the test workspace has no ``.secrets.yaml``."""
    return []


@pytest.fixture(autouse=True)
def fake_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate the test in a temporary workspace with the fixture ecosystem file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(workspace))
    monkeypatch.setenv(ECOSYSTEM_FILE_ENV, str(ECOSYSTEM_FIXTURE))
    monkeypatch.delenv(task_resolution.TASK_TAXONOMY_ENV, raising=False)
    monkeypatch.setattr(secrets_vault, "load_secrets_records", _no_vault_records)
    for name in TOKEN_SECRET_VARS:
        # setenv records the original state, so the real value comes back afterwards.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    ecosystem.clear_cache()
    task_resolution.clear_cache()
    yield workspace
    ecosystem.clear_cache()
    task_resolution.clear_cache()


KAGGLE_CREDENTIAL_VARS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")
"""Environment variables the Kaggle credential resolver reads and exports."""


@pytest.fixture(name="kaggle_home")
def fixture_kaggle_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder (``Path.home()``) and no Kaggle variables in the environment."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for name in KAGGLE_CREDENTIAL_VARS:
        # setenv records the original state, so exported credentials are undone afterwards.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    return home
