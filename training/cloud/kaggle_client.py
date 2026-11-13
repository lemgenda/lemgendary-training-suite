"""Authenticated Kaggle API clients and a real (network) authentication probe.

The credentials themselves come from the one resolver,
:func:`training.cloud.credentials.resolve_kaggle_credentials`. This module turns a resolved
``(username, token)`` pair into an authenticated ``KaggleApi`` and checks it against Kaggle.

Token forms (kaggle >= 2.x / kagglesdk):

* ``KGAT_...`` access tokens are passed unchanged through ``KAGGLE_API_TOKEN``; the SDK reads
  that variable first, then ``~/.kaggle/access_token`` (and ``access_token.txt``).
* Legacy API keys use ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (or ``~/.kaggle/kaggle.json``).

``KaggleApi.authenticate()`` succeeds as soon as a legacy username and key exist, without a
network call, and calls ``exit(1)`` when no credentials exist at all. :func:`probe_kaggle_auth`
therefore makes one cheap authenticated request (list the caller's own kernels, page size 1)
and reports its real outcome, and :func:`authenticated_kaggle_api` turns the SDK's
``SystemExit`` into :class:`KaggleAuthError`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import requests

from training.cloud.credentials import export_kaggle_credentials, is_kaggle_access_token
from training.cloud.errors import KAGGLE_API_ERRORS
from training.utils.lazy_imports import load_kaggle_api

logger = logging.getLogger("lemtrain.cloud.kaggle_client")

AUTH_METHOD_ACCESS_TOKEN = "access_token"
AUTH_METHOD_LEGACY_KEY = "legacy_api_key"


class KaggleAuthError(RuntimeError):
    """The Kaggle SDK could not be loaded or rejected the credentials."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def _http_status(exc: BaseException) -> int | None:
    """HTTP status code carried by a ``requests`` error, if any."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def auth_method(token: str) -> str:
    """How the SDK will authenticate ``token`` (access token or legacy API key)."""
    return AUTH_METHOD_ACCESS_TOKEN if is_kaggle_access_token(token) else AUTH_METHOD_LEGACY_KEY


def authenticated_kaggle_api(username: str, token: str) -> Any:
    """Return a ``KaggleApi`` authenticated with ``username`` / ``token``.

    Raises :class:`KaggleAuthError` when the SDK is missing, rejects the token, or finds no
    credentials (the SDK's ``exit(1)`` is caught here so a server job thread never dies).
    """
    export_kaggle_credentials(username, token)
    try:
        api = load_kaggle_api()()
        api.authenticate()
    except SystemExit as exc:
        logger.warning("Kaggle SDK found no usable credentials for '%s' (exit %s)", username, exc)
        raise KaggleAuthError(f"No usable Kaggle credentials for '{username}'") from exc
    except ImportError as exc:
        raise KaggleAuthError(f"Kaggle SDK is not installed: {exc}") from exc
    except KAGGLE_API_ERRORS as exc:
        raise KaggleAuthError(
            f"Kaggle authentication failed for '{username}': {exc}", _http_status(exc)
        ) from exc
    return api


@dataclass(frozen=True)
class KaggleAuthStatus:
    """Outcome of a real authenticated Kaggle call."""

    authenticated: bool
    username: str
    token_configured: bool
    auth_method: str | None = None
    http_status: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the status route."""
        return asdict(self)


ApiFactory = Callable[[str, str], Any]


def probe_kaggle_auth(
    username: str, token: str, api_factory: ApiFactory = authenticated_kaggle_api
) -> KaggleAuthStatus:
    """Authenticate and list the caller's own kernels (page size 1); report what happened."""
    if not (username and token):
        return KaggleAuthStatus(
            authenticated=False,
            username=username,
            token_configured=bool(token),
            error="Kaggle username or token is not configured",
        )
    method = auth_method(token)
    try:
        api = api_factory(username, token)
        api.kernels_list(mine=True, page_size=1)
    except KaggleAuthError as exc:
        return KaggleAuthStatus(False, username, True, method, exc.http_status, str(exc))
    except KAGGLE_API_ERRORS as exc:
        status = _http_status(exc)
        logger.warning("Kaggle auth probe failed for %s (HTTP %s): %s", username, status, exc)
        return KaggleAuthStatus(False, username, True, method, status, str(exc))
    return KaggleAuthStatus(True, username, True, method, 200)
