"""Unified cloud credentials discovery, parsing, and stealth masking."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from training.config import secrets as secrets_vault
from training.config.secrets import get_secret
from training.ecosystem import kaggle_owner

logger = logging.getLogger("lemtrain.cloud.credentials")


def mask_secret(text: str, secret: str | None) -> str:
    """Replace occurrences of secret in text with masked stealth placeholder."""
    if not secret or len(secret) < 4:
        return text
    return text.replace(secret, "***STEALTH***")


def resolve_github_credentials(override_pat: str | None = None) -> str | None:
    """Resolve GitHub Personal Access Token with fallback order.

    1. Explicit override parameter
    2. Environment variables: GITHUB_PAT, SUITE_PAT
    3. The default ``github`` record of the ``.secrets.yaml`` vault (``get_secret``)
    """
    if override_pat and override_pat.strip():
        return override_pat.strip()

    token = get_secret("GITHUB_PAT") or get_secret("SUITE_PAT")
    if token and token.strip():
        return token.strip()

    return None


def load_secrets_yaml_registry() -> list[dict[str, Any]]:
    """Records of the workspace ``.secrets.yaml`` vault (the one vault reader, SSOT)."""
    return secrets_vault.load_secrets_records()


def load_kaggle_users() -> list[dict[str, Any]]:
    """Kaggle accounts of the ``.secrets.yaml`` vault (default account first).

    The vault is the only account registry; the legacy ``.kaggle_users`` and ``.kaggle_token``
    files are no longer read (K-AUTH-4).
    """
    accounts: list[dict[str, Any]] = []
    secrets = load_secrets_yaml_registry()
    if secrets:
        kaggle_secrets = [s for s in secrets if str(s.get("service", "")).lower() == "kaggle"]
        if kaggle_secrets:
            sorted_secrets = sorted(
                kaggle_secrets, key=lambda s: not bool(s.get("is_default", False))
            )
            for s in sorted_secrets:
                u = s.get("username") or ""
                t = s.get("secret_value") or ""
                if u and t:
                    accounts.append(
                        {
                            "username": u,
                            "token": t,
                            "is_default": bool(s.get("is_default", False)),
                        }
                    )
            if accounts:
                return accounts

    return accounts


def _fill_from_kaggle_json(k_user: str, k_key: str) -> tuple[str, str]:
    """Fill missing values from ~/.kaggle/kaggle.json (utf-8-sig, utf-8, utf-16, latin-1)."""
    user_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    try:
        if not user_kaggle_json.exists():
            return k_user, k_key
        for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
            try:
                creds = json.loads(user_kaggle_json.read_text(encoding=enc))
            except (OSError, ValueError) as e:
                logger.debug("Failed parsing ~/.kaggle/kaggle.json with %s: %s", enc, e)
                continue
            if not k_user:
                k_user = creds.get("username", "")
            if not k_key:
                k_key = creds.get("key", "")
            break
    except OSError as e:
        logger.debug("Could not access ~/.kaggle/kaggle.json: %s", e)
    return k_user, k_key


KAGGLE_ACCESS_TOKEN_PREFIX = "KGAT_"
"""Prefix of Kaggle access tokens (sent unchanged through ``KAGGLE_API_TOKEN``)."""

_ACCESS_TOKEN_FILES = ("access_token", "access_token.txt")
"""Files under ``~/.kaggle`` the Kaggle SDK reads an access token from, in its order."""


def is_kaggle_access_token(token: str) -> bool:
    """True for a ``KGAT_`` access token, False for a legacy API key."""
    return token.strip().startswith(KAGGLE_ACCESS_TOKEN_PREFIX)


def _read_token_file(path: Path) -> str:
    """Return the stripped token in ``path`` ('' when absent or unreadable)."""
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.debug("Could not read Kaggle token file %s: %s", path, e)
        return ""


def _env_token() -> str:
    """``KAGGLE_API_TOKEN`` (a token, or a path to a token file, as the SDK accepts)."""
    raw = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if raw and Path(raw).is_file():
        return _read_token_file(Path(raw))
    return raw


def _token_file_key(k_key: str) -> str:
    """Fall back to ``~/.kaggle/access_token`` (or ``access_token.txt``).

    Tokens are returned unchanged: a ``KGAT_`` token is an access token, never a legacy key.
    """
    candidates = [Path.home() / ".kaggle" / name for name in _ACCESS_TOKEN_FILES]
    for candidate in candidates:
        if k_key:
            break
        k_key = _read_token_file(candidate)
    return k_key


def export_kaggle_credentials(k_user: str, k_key: str) -> None:
    """Export the credentials the Kaggle SDK, kagglehub and child processes read.

    A ``KGAT_`` access token goes unchanged into ``KAGGLE_API_TOKEN`` (read first by the SDK);
    a stale ``KGAT_`` value in ``KAGGLE_KEY`` is removed because the SDK would send it as a
    legacy key. A legacy key goes into ``KAGGLE_KEY`` and clears ``KAGGLE_API_TOKEN``, which
    would otherwise win over it. No credential file is written.
    """
    if k_user:
        os.environ["KAGGLE_USERNAME"] = k_user
    if not k_key:
        return
    if is_kaggle_access_token(k_key):
        os.environ["KAGGLE_API_TOKEN"] = k_key.strip()
        if is_kaggle_access_token(os.environ.get("KAGGLE_KEY", "")):
            os.environ.pop("KAGGLE_KEY", None)
    else:
        os.environ["KAGGLE_KEY"] = k_key
        os.environ.pop("KAGGLE_API_TOKEN", None)


def _vault_account(accounts: list[dict[str, Any]], user: str) -> dict[str, Any] | None:
    """The vault entry of ``user`` (the default account when ``user`` is empty)."""
    if not accounts:
        return None
    if user:
        return next((a for a in accounts if a.get("username") == user), None)
    return next((a for a in accounts if a.get("is_default")), accounts[0])


def resolve_kaggle_credentials(
    override_user: str | None = None,
    override_key: str | None = None,
) -> tuple[str, str]:
    """Resolve Kaggle credentials (the one resolver of the suite) and export them.

    1. Explicit parameters
    2. Ecosystem .secrets.yaml vault (default active Kaggle secret)
    3. Environment variables (``KAGGLE_USERNAME``; ``KAGGLE_API_TOKEN``, then ``KAGGLE_KEY``)
    4. User home ``~/.kaggle/kaggle.json``
    5. ``~/.kaggle/access_token`` (or ``access_token.txt``)
    6. The ecosystem Kaggle owner (``identity.kaggle_owner`` in ``ecosystem.json``)

    A vault token is taken only together with its own account: an ``override_user`` without a
    key uses that user's vault entry, never the default account's token (K-AUTH-4).

    The token is returned exactly as stored (``KGAT_`` prefix kept). Nothing is written to
    disk; see :func:`export_kaggle_credentials` for the environment variables set.
    """
    k_user = override_user or ""
    k_key = override_key or ""

    # Ecosystem .secrets.yaml registry check
    if not k_key:
        account = _vault_account(load_kaggle_users(), k_user)
        if account is not None:
            k_user = k_user or str(account.get("username", ""))
            k_key = str(account.get("token", ""))

    # Environment variables fallback
    if not k_user:
        k_user = os.environ.get("KAGGLE_USERNAME", "")
    if not k_key:
        k_key = _env_token() or os.environ.get("KAGGLE_KEY", "")

    if not (k_user and k_key):
        k_user, k_key = _fill_from_kaggle_json(k_user, k_key)
    k_key = _token_file_key(k_key)

    if not k_user:
        k_user = kaggle_owner()

    export_kaggle_credentials(k_user, k_key)
    return k_user, k_key


def resolve_gdrive_credentials(override_token: str | None = None) -> str | None:
    """Resolve Google Drive access credentials using precedence.

    1. Explicit override token
    2. Environment variable GOOGLE_DRIVE or GOOGLE_DRIVE_JSON
    3. The default ``google_drive`` record of the ``.secrets.yaml`` vault (``get_secret``)
    """
    if override_token and override_token.strip():
        return override_token.strip()

    token = get_secret("GOOGLE_DRIVE")
    if token and token.strip():
        return token.strip()

    json_cred = get_secret("GOOGLE_DRIVE_JSON")
    if json_cred and json_cred.strip():
        return json_cred.strip()

    return None
