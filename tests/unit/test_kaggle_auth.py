"""The one Kaggle credential resolver, the authenticated client and the real status probe.

Every test uses fakes: no Kaggle SDK call reaches the network and the home folder is a
temporary directory, so no credential file of the machine is read or written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from training.cloud import credentials, kaggle_client
from training.cloud.credentials import export_kaggle_credentials, resolve_kaggle_credentials
from training.cloud.kaggle_client import (
    AUTH_METHOD_ACCESS_TOKEN,
    AUTH_METHOD_LEGACY_KEY,
    KaggleAuthError,
    authenticated_kaggle_api,
    probe_kaggle_auth,
)
from training.server.app import create_app

_ACCESS = "KGAT_fake0123456789"
_LEGACY = "0123456789abcdef0123456789abcdef"
_KAGGLE_VARS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")


@pytest.fixture(autouse=True, name="isolated_home")
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home folder and no Kaggle variables in the environment."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for name in _KAGGLE_VARS:
        # setenv records the original state, so exported credentials are undone afterwards.
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    monkeypatch.setattr(credentials, "load_kaggle_users", lambda users_file=None: [])
    return home


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=response)


class _FakeApi:
    """A fake ``KaggleApi``: raises the given errors and records ``kernels_list`` calls."""

    def __init__(
        self, error: BaseException | None = None, auth_error: BaseException | None = None
    ) -> None:
        self.error = error
        self.auth_error = auth_error
        self.calls: list[dict[str, Any]] = []

    def authenticate(self) -> None:
        """Raise ``auth_error`` when set."""
        if self.auth_error is not None:
            raise self.auth_error

    def kernels_list(self, **kwargs: Any) -> list[Any]:
        """Record the call; raise ``error`` when set."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return []


def _sdk_returning(api: _FakeApi) -> Any:
    """A ``load_kaggle_api`` replacement whose ``KaggleApi()`` is ``api``."""
    return lambda: lambda: api


class TestResolver:
    """``resolve_kaggle_credentials`` keeps tokens unchanged and writes no file."""

    def test_access_token_passes_unchanged(self, isolated_home: Path) -> None:
        """A KGAT_ token goes unchanged into KAGGLE_API_TOKEN; no file is written."""
        user, token = resolve_kaggle_credentials("someone", _ACCESS)
        assert (user, token) == ("someone", _ACCESS)
        assert os.environ["KAGGLE_API_TOKEN"] == _ACCESS
        assert "KAGGLE_KEY" not in os.environ
        assert not (isolated_home / ".kaggle").exists()

    def test_stale_access_token_in_legacy_key_is_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A KGAT_ value left in KAGGLE_KEY would be sent as a legacy key."""
        monkeypatch.setenv("KAGGLE_KEY", _ACCESS)
        export_kaggle_credentials("someone", _ACCESS)
        assert "KAGGLE_KEY" not in os.environ

    def test_legacy_key_clears_access_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A legacy key wins over an ambient access token."""
        monkeypatch.setenv("KAGGLE_API_TOKEN", _ACCESS)
        user, token = resolve_kaggle_credentials("someone", _LEGACY)
        assert (user, token) == ("someone", _LEGACY)
        assert os.environ["KAGGLE_KEY"] == _LEGACY
        assert "KAGGLE_API_TOKEN" not in os.environ

    def test_env_access_token_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KAGGLE_API_TOKEN from the environment is a credential source."""
        monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
        monkeypatch.setenv("KAGGLE_API_TOKEN", _ACCESS)
        assert resolve_kaggle_credentials() == ("envuser", _ACCESS)

    def test_access_token_file_is_read_unchanged(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """~/.kaggle/access_token is read unchanged and kaggle.json is not created."""
        monkeypatch.setenv("KAGGLE_USERNAME", "fileuser")
        token_file = isolated_home / ".kaggle" / "access_token"
        token_file.parent.mkdir()
        token_file.write_text(_ACCESS + "\n", encoding="utf-8")
        assert resolve_kaggle_credentials() == ("fileuser", _ACCESS)
        assert sorted(p.name for p in token_file.parent.iterdir()) == ["access_token"]


class TestAuthenticatedApi:
    """``authenticated_kaggle_api`` never lets the SDK exit the process."""

    def test_sdk_exit_becomes_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The SDK's ``exit(1)`` (no credentials) is raised as ``KaggleAuthError``."""
        fake = _FakeApi(auth_error=SystemExit(1))
        monkeypatch.setattr(kaggle_client, "load_kaggle_api", _sdk_returning(fake))
        with pytest.raises(KaggleAuthError):
            authenticated_kaggle_api("someone", _ACCESS)

    def test_http_error_keeps_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rejected token keeps its HTTP status."""
        fake = _FakeApi(auth_error=_http_error(401))
        monkeypatch.setattr(kaggle_client, "load_kaggle_api", _sdk_returning(fake))
        with pytest.raises(KaggleAuthError) as info:
            authenticated_kaggle_api("someone", _ACCESS)
        assert info.value.http_status == 401


class TestProbe:
    """``probe_kaggle_auth`` reports the outcome of a real authenticated call."""

    def test_success_lists_own_kernels_once(self) -> None:
        """A working token lists the caller's own kernels once (page size 1)."""
        api = _FakeApi()
        status = probe_kaggle_auth("someone", _ACCESS, lambda u, t: api)
        assert status.authenticated
        assert status.auth_method == AUTH_METHOD_ACCESS_TOKEN
        assert api.calls == [{"mine": True, "page_size": 1}]

    def test_rejected_call_is_not_authenticated(self) -> None:
        """A 401 from the call means not authenticated."""
        status = probe_kaggle_auth("someone", _LEGACY, lambda u, t: _FakeApi(_http_error(401)))
        assert not status.authenticated
        assert status.http_status == 401
        assert status.auth_method == AUTH_METHOD_LEGACY_KEY
        assert status.error

    def test_auth_error_is_reported(self) -> None:
        """A failed authentication is reported with its message."""

        def _factory(user: str, token: str) -> Any:
            raise KaggleAuthError("no credentials")

        status = probe_kaggle_auth("someone", _ACCESS, _factory)
        assert not status.authenticated
        assert status.error == "no credentials"

    def test_missing_token_makes_no_call(self) -> None:
        """Without a token nothing is called."""

        def _factory(user: str, token: str) -> Any:
            raise AssertionError("must not be called")

        status = probe_kaggle_auth("someone", "", _factory)
        assert not status.authenticated
        assert not status.token_configured


def test_status_route_reports_probe_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GET /kaggle/status`` returns the probe result, not a configured-key guess."""
    fake = _FakeApi(_http_error(401))
    monkeypatch.setattr(kaggle_client, "load_kaggle_api", _sdk_returning(fake))
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(project_root=Path(tmp), enforce_auth=False)
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        monkeypatch.setenv("KAGGLE_API_TOKEN", _ACCESS)
        with TestClient(app) as client:
            body = client.get("/api/training/kaggle/status").json()
    assert body["authenticated"] is False
    assert body["http_status"] == 401
    assert body["username"] == "someone"
    assert body["token_configured"] is True
    assert fake.calls == [{"mine": True, "page_size": 1}]
