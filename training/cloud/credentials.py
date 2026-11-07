"""Unified cloud credentials discovery, parsing, and stealth masking."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from training.config.secrets import get_secret
from training.ecosystem import kaggle_owner
from training.utils.paths import get_project_root, get_workspace_root

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
    3. Structured secrets loaded from dotfiles
    """
    if override_pat and override_pat.strip():
        return override_pat.strip()

    token = get_secret("GITHUB_PAT") or get_secret("SUITE_PAT")
    if token and token.strip():
        return token.strip()

    return None


def load_secrets_yaml_registry() -> list[dict[str, Any]]:
    """Load secrets registry from workspace .secrets.yaml."""
    candidates = [
        get_workspace_root() / ".secrets.yaml",
        get_project_root() / ".secrets.yaml",
        get_project_root().parent / ".secrets.yaml",
        Path.cwd() / ".secrets.yaml",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                return data.get("secrets", [])
            except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as e:
                logger.warning("Failed parsing .secrets.yaml at %s: %s", p, e)
    return []


def find_kaggle_users_file(explicit_path: Path | None = None) -> Path | None:
    """Locate the .kaggle_users registry file across project hierarchies."""
    if explicit_path and explicit_path.exists() and explicit_path.is_file():
        return explicit_path

    project_root = get_project_root()
    candidates = [
        Path.cwd() / ".kaggle_users",
        project_root / ".kaggle_users",
        project_root.parent / ".kaggle_users",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    default_target = project_root / ".kaggle_users"
    return default_target if default_target.exists() else None


def load_kaggle_users(users_file: Path | None = None) -> list[dict[str, Any]]:
    """Parse configured Kaggle user accounts from .secrets.yaml."""
    accounts: list[dict[str, Any]] = []

    # 1. Primary: load from .secrets.yaml
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

    # 2. Fallback: parse .kaggle_users dotfile if provided or present
    target_file = find_kaggle_users_file(users_file)
    if target_file and target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
            owner = kaggle_owner()
            accounts.extend(
                {"username": user, "token": token, "is_default": user == owner}
                for user, token in parse_kaggle_users_text(content)
            )
            accounts.sort(key=lambda a: not a.get("is_default", False))
        except OSError as e:
            logger.warning("Could not read .kaggle_users at %s: %s", target_file, e)

    return accounts


def parse_kaggle_users_text(content: str) -> list[tuple[str, str]]:
    """Parse ``.kaggle_users`` text into (username, token) pairs.

    Expected format per line: ``KAGGLE_USERNAME=<user>, KAGGLE_API_TOKEN=<token>;``.
    Blank lines and ``#`` comments are skipped.
    """
    pairs: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m_user = re.search(r"KAGGLE_USERNAME=([^,;\s]+)", line)
        m_token = re.search(r"KAGGLE_API_TOKEN=([^,;\s]+)", line)
        if m_user and m_token:
            pairs.append((m_user.group(1).strip(), m_token.group(1).strip()))
    return pairs


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


def _token_file_key(k_key: str) -> str:
    """Fall back to ~/.kaggle/access_token, then the project ``.kaggle_token`` (KGAT_ removed)."""
    access_token_file = Path.home() / ".kaggle" / "access_token"
    if not k_key and access_token_file.exists():
        try:
            tok = access_token_file.read_text(encoding="utf-8").strip()
            if tok:
                k_key = tok
        except OSError as e:
            logger.debug("Could not read ~/.kaggle/access_token: %s", e)

    token_file = get_project_root() / ".kaggle_token"
    if not k_key and token_file.exists():
        try:
            raw = token_file.read_text(encoding="utf-8").strip()
            if raw.startswith("KGAT_"):
                raw = raw.replace("KGAT_", "")
            k_key = raw
        except OSError as e:
            logger.debug("Failed reading .kaggle_token: %s", e)
    return k_key


def _export_kaggle_env(k_user: str, k_key: str) -> None:
    """Export the credentials for the Kaggle Python SDK and child processes."""
    if k_user:
        os.environ["KAGGLE_USERNAME"] = k_user
    if k_key:
        os.environ["KAGGLE_KEY"] = k_key
        if k_key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = k_key
        else:
            # If a legacy key is active, clear any conflicting access token in env
            os.environ.pop("KAGGLE_API_TOKEN", None)


def _sync_kaggle_json(k_user: str, k_key: str) -> None:
    """Write ~/.kaggle/kaggle.json in clean UTF-8 so the Kaggle CLI works seamlessly."""
    try:
        k_json = Path.home() / ".kaggle" / "kaggle.json"
        k_json.parent.mkdir(parents=True, exist_ok=True)
        k_content = json.dumps({"username": k_user, "key": k_key}, indent=2)
        k_json.write_text(k_content, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not sync Kaggle CLI credentials file %s: %s", k_json, exc)


def resolve_kaggle_credentials(
    override_user: str | None = None,
    override_key: str | None = None,
) -> tuple[str, str]:
    """Resolve Kaggle credentials with senior fallback hierarchy.

    1. Explicit parameters
    2. Ecosystem .secrets.yaml vault (default active Kaggle secret)
    3. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
    4. User home ~/.kaggle/kaggle.json
    5. Sibling/project .kaggle_token or .kaggle_users
    6. The ecosystem Kaggle owner (``identity.kaggle_owner`` in ``ecosystem.json``)
    """
    k_user = override_user or ""
    k_key = override_key or ""

    # Ecosystem .secrets.yaml registry check
    if not (k_user and k_key):
        accounts = load_kaggle_users()
        if accounts:
            default_acc = next((a for a in accounts if a.get("is_default")), accounts[0])
            if not k_user:
                k_user = default_acc.get("username", "")
            if not k_key:
                k_key = default_acc.get("token", "")

    # Environment variables fallback
    if not k_user:
        k_user = os.environ.get("KAGGLE_USERNAME", "")
    if not k_key:
        k_key = os.environ.get("KAGGLE_KEY", "")

    if not (k_user and k_key):
        k_user, k_key = _fill_from_kaggle_json(k_user, k_key)
    k_key = _token_file_key(k_key)

    if not k_user:
        k_user = kaggle_owner()

    _export_kaggle_env(k_user, k_key)
    if k_user and k_key:
        _sync_kaggle_json(k_user, k_key)

    return k_user, k_key


def resolve_gdrive_credentials(override_token: str | None = None) -> str | None:
    """Resolve Google Drive access credentials using precedence.

    1. Explicit override token
    2. Local file .GOOGLE_DRIVE or .GOOGLE_DRIVE_JSON
    3. Environment variable GOOGLE_DRIVE or GOOGLE_DRIVE_JSON
    4. Structured secrets from config loader
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
