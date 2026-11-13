"""Unit tests for training/config/secrets.py in LemGendary Model Training Suite.

The global conftest stubs the vault reader and unsets the token variables; the tests here
check that isolation and exercise the real file lookup against a vault in the fake workspace.
"""

from pathlib import Path

import pytest

from training.cloud.credentials import (
    load_secrets_yaml_registry,
    resolve_gdrive_credentials,
    resolve_github_credentials,
)
from training.config import secrets as secrets_vault
from training.config.secrets import _mask_secret, find_secrets_yaml, get_secret, load_secrets
from training.config.secrets import load_secrets_records as real_vault_reader

_VAULT = """secrets:
  - service: github
    secret_value: ghp_fixture_token
    is_default: true
  - service: google_drive
    secret_value: gdrive_fixture_token
"""


def test_mask_secret() -> None:
    """Secret masking for empty, short and long values."""
    assert _mask_secret("") == ""
    assert _mask_secret("short") == "***"
    assert _mask_secret("ghp_1234567890abcdef") == "ghp...def"


def test_load_secrets_reports_unset_tokens() -> None:
    """With the stubbed vault every known key is reported, the tokens as not set."""
    summary = load_secrets()
    assert summary["GITHUB_PAT"] == "[NOT_SET]"
    assert summary["GOOGLE_DRIVE"] == "[NOT_SET]"
    assert "KAGGLE_KEY" in summary


def test_tokens_never_come_from_the_real_vault() -> None:
    """GitHub and Google Drive lookups see no vault and no inherited variables."""
    assert secrets_vault.load_secrets_records() == []
    assert load_secrets_yaml_registry() == []
    assert get_secret("GITHUB_PAT") is None
    assert resolve_github_credentials() is None
    assert resolve_gdrive_credentials() is None


def test_get_secret_default() -> None:
    """get_secret falls back to the default."""
    assert (
        get_secret("NON_EXISTENT_LEMGENDARY_SECRET_KEY", default="fallback_val") == "fallback_val"
    )


def test_get_secret_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_secret reads (and strips) the environment."""
    monkeypatch.setenv("GITHUB_PAT", " my_secret_token_123 ")
    assert get_secret("GITHUB_PAT") == "my_secret_token_123"
    assert resolve_github_credentials() == "my_secret_token_123"


def test_override_beats_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit override token wins over the environment."""
    monkeypatch.setenv("GOOGLE_DRIVE", "env_token")
    assert resolve_gdrive_credentials(" override ") == "override"
    assert resolve_gdrive_credentials() == "env_token"


def test_vault_lookup_prefers_the_workspace_file(fake_workspace: Path) -> None:
    """The real reader (bound before the stub) parses a vault in the fake workspace root."""
    vault = fake_workspace / ".secrets.yaml"
    vault.write_text(_VAULT, encoding="utf-8")
    assert find_secrets_yaml() == vault
    services = [record["service"] for record in real_vault_reader()]
    assert services == ["github", "google_drive"]
