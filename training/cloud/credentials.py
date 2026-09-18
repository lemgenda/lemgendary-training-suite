"""Unified cloud credentials discovery, parsing, and stealth masking."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from training.config.secrets import get_secret, load_secrets
from training.utils.paths import get_project_root

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


def load_kaggle_users(users_file: Path | None = None) -> list[dict[str, str]]:
    """Parse configured Kaggle user accounts from .kaggle_users.

    Expected format per line: KAGGLE_USERNAME=<user>, KAGGLE_API_TOKEN=<token>;
    """
    target_file = find_kaggle_users_file(users_file)
    accounts: list[dict[str, str]] = []

    if target_file and target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m_user = re.search(r"KAGGLE_USERNAME=([^,;\s]+)", line)
                m_token = re.search(r"KAGGLE_API_TOKEN=([^,;\s]+)", line)
                if m_user and m_token:
                    accounts.append({
                        "username": m_user.group(1).strip(),
                        "token": m_token.group(1).strip(),
                    })
        except OSError as e:
            logger.warning("Could not read .kaggle_users at %s: %s", target_file, e)

    return accounts


def resolve_kaggle_credentials(
    override_user: str | None = None,
    override_key: str | None = None,
) -> tuple[str, str]:
    """Resolve Kaggle credentials with senior fallback hierarchy.

    1. Explicit parameters
    2. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
    3. User home ~/.kaggle/kaggle.json
    4. Sibling/project .kaggle_token or .kaggle_users
    """
    k_user = override_user or os.environ.get("KAGGLE_USERNAME", "")
    k_key = override_key or os.environ.get("KAGGLE_KEY", "")

    user_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not (k_user and k_key) and user_kaggle_json.exists():
        try:
            creds = json.loads(user_kaggle_json.read_text(encoding="utf-8"))
            if not k_user:
                k_user = creds.get("username", "")
            if not k_key:
                k_key = creds.get("key", "")
        except (OSError, ValueError) as e:
            logger.debug("Failed parsing ~/.kaggle/kaggle.json: %s", e)

    # Project .kaggle_token fallback
    project_root = get_project_root()
    token_file = project_root / ".kaggle_token"
    if not k_key and token_file.exists():
        try:
            raw = token_file.read_text(encoding="utf-8").strip()
            if raw.startswith("KGAT_"):
                raw = raw.replace("KGAT_", "")
            k_key = raw
        except OSError as e:
            logger.debug("Failed reading .kaggle_token: %s", e)

    # Multi-account registry fallback
    if not (k_user and k_key):
        accounts = load_kaggle_users()
        if accounts:
            first = accounts[0]
            if not k_user:
                k_user = first.get("username", "")
            if not k_key:
                k_key = first.get("token", "")

    if not k_user:
        k_user = "lemtreursi"

    if k_user:
        os.environ["KAGGLE_USERNAME"] = k_user
    if k_key:
        os.environ["KAGGLE_KEY"] = k_key

    return k_user, k_key


def resolve_gdrive_credentials(override_token: str | None = None) -> str | None:
    """Resolve Google Drive access credentials using precedence.

    1. Explicit override token
    2. Local file .GOOGLE_DRIVE
    3. Environment variable GOOGLE_DRIVE
    4. Structured secrets from config loader
    """
    if override_token and override_token.strip():
        return override_token.strip()

    token = get_secret("GOOGLE_DRIVE")
    if token and token.strip():
        return token.strip()

    return None

