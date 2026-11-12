"""Legacy Kaggle credential files are gone; the secrets cell exports a KGAT_ token only.

R3-K3 item 3 (K-AUTH-3, K-AUTH-4, K-NB-6). The home folder is a temporary directory and the
platform secrets clients are fakes, so no real credential or file is read or written.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from training import kaggle_accounts
from training.cloud import credentials
from training.cloud.credentials import resolve_kaggle_credentials
from training.kaggle_accounts import vault_accounts
from training.notebooks.cells.secrets import build_secrets_cell

_ACCESS = "KGAT_fake0123456789"
_OTHER = "KGAT_other0123456789"
_ACCOUNTS = [
    {"username": "default-user", "token": _ACCESS, "is_default": True},
    {"username": "second-user", "token": _OTHER, "is_default": False},
]


@pytest.fixture(autouse=True, name="isolated_home")
def _isolated_home(kaggle_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder, the fixture vault and no Kaggle variables in the environment."""
    monkeypatch.setattr(credentials, "load_kaggle_users", lambda: [dict(a) for a in _ACCOUNTS])
    monkeypatch.setattr(kaggle_accounts, "load_kaggle_users", lambda: [dict(a) for a in _ACCOUNTS])
    monkeypatch.setattr(credentials, "load_secrets_yaml_registry", list)
    return kaggle_home


def test_override_user_never_takes_another_accounts_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user without a key gets its own vault token; an unknown user gets none from the vault."""
    assert resolve_kaggle_credentials("second-user") == ("second-user", _OTHER)
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    assert resolve_kaggle_credentials("stranger") == ("stranger", "")


def test_default_account_without_override() -> None:
    """Without an override the vault's default account is used."""
    assert resolve_kaggle_credentials() == ("default-user", _ACCESS)


def test_legacy_registry_files_are_not_supported() -> None:
    """The ``.kaggle_users`` and ``.kaggle_token`` readers are deleted."""
    source = Path(credentials.__file__).read_text(encoding="utf-8")
    assert '/ ".kaggle_token"' not in source
    assert '/ ".kaggle_users"' not in source
    assert not hasattr(credentials, "find_kaggle_users_file")
    assert vault_accounts() == [
        {"username": "default-user", "token": _ACCESS},
        {"username": "second-user", "token": _OTHER},
    ]


class _BackendError(Exception):
    """Stand-in for ``kaggle_web_client.BackendError`` (secret not attached)."""


def _kaggle_modules(secrets: dict[str, str]) -> dict[str, ModuleType]:
    """Fake ``kaggle_secrets`` / ``kaggle_web_client`` modules serving ``secrets``."""

    def get_secret(label: str) -> str:
        """Return the secret or raise like the real client."""
        if label not in secrets:
            raise _BackendError(label)
        return secrets[label]

    secrets_mod = ModuleType("kaggle_secrets")
    vars(secrets_mod)["UserSecretsClient"] = lambda: SimpleNamespace(get_secret=get_secret)
    web_mod = ModuleType("kaggle_web_client")
    vars(web_mod)["BackendError"] = _BackendError
    return {"kaggle_secrets": secrets_mod, "kaggle_web_client": web_mod}


def _run_cell(target: str, folder: Path) -> None:
    """Run the generated secrets cell of ``target`` as a script."""
    script = folder / f"{target}_secrets_cell.py"
    script.write_text("".join(build_secrets_cell(target)["source"]), encoding="utf-8")
    runpy.run_path(str(script))


def _run_kaggle_cell(
    monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str], folder: Path
) -> None:
    for name, module in _kaggle_modules(secrets).items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("SUITE_PAT", "GITHUB_PAT", "GOOGLE_DRIVE"):
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    _run_cell("kaggle", folder)


def test_kaggle_cell_exports_access_token_only(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path, tmp_path: Path
) -> None:
    """A KGAT_ secret goes to KAGGLE_API_TOKEN; no Kaggle credential file is written."""
    _run_kaggle_cell(monkeypatch, {"KAGGLE_API_TOKEN": _ACCESS, "SUITE_PAT": "pat"}, tmp_path)
    assert os.environ["KAGGLE_API_TOKEN"] == _ACCESS
    assert "KAGGLE_KEY" not in os.environ
    assert os.environ["SUITE_PAT"] == "pat"
    assert not (isolated_home / ".kaggle").exists()


def test_kaggle_cell_rejects_legacy_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A legacy key in the secret is not exported."""
    _run_kaggle_cell(monkeypatch, {"KAGGLE_API_TOKEN": "0123456789abcdef"}, tmp_path)
    assert "KAGGLE_API_TOKEN" not in os.environ
    assert "legacy keys are not used" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["kaggle", "colab"])
def test_secrets_cell_has_no_obfuscation_or_broad_except(target: str) -> None:
    """Plain imports, no ``kaggle.json`` and only the client's own errors are caught."""
    source = "".join(build_secrets_cell(target)["source"])
    compile(source, f"{target}-secrets", "exec")
    for banned in ("b64decode", "__import__", "kaggle.json", "except Exception", "pass\n"):
        assert banned not in source
    expected: Any = (
        "from kaggle_secrets import UserSecretsClient"
        if target == "kaggle"
        else "from google.colab import userdata"
    )
    assert expected in source


def test_colab_cell_reads_through_userdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Colab cell reads secrets with ``userdata.get`` and the same token rule."""

    class SecretNotFoundError(Exception):
        """Fake missing-secret error."""

    class NotebookAccessError(Exception):
        """Fake access-denied error."""

    def get(name: str) -> str:
        if name != "KAGGLE_API_TOKEN":
            raise SecretNotFoundError(name)
        return _ACCESS

    userdata = SimpleNamespace(
        get=get, SecretNotFoundError=SecretNotFoundError, NotebookAccessError=NotebookAccessError
    )
    colab = ModuleType("google.colab")
    vars(colab)["userdata"] = userdata
    google = ModuleType("google")
    vars(google)["colab"] = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _run_cell("colab", tmp_path)
    assert os.environ["KAGGLE_API_TOKEN"] == _ACCESS
