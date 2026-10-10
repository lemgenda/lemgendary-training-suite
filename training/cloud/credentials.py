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


def load_secrets_yaml_registry() -> list[dict[str, Any]]:
    """Load secrets registry from workspace .secrets.yaml."""
    from training.utils.paths import get_workspace_root
    candidates = [
        get_workspace_root() / ".secrets.yaml",
        get_project_root() / ".secrets.yaml",
        get_project_root().parent / ".secrets.yaml",
        Path.cwd() / ".secrets.yaml",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                return data.get("secrets", [])
            except Exception as e:
                logger.debug("Failed parsing .secrets.yaml at %s: %s", p, e)
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
    """Parse configured Kaggle user accounts, prioritizing default from .secrets.yaml."""
    accounts: list[dict[str, Any]] = []

    # 1. Primary: load from .secrets.yaml
    secrets = load_secrets_yaml_registry()
    if secrets:
        kaggle_secrets = [s for s in secrets if str(s.get("service", "")).lower() == "kaggle"]
        if kaggle_secrets:
            sorted_secrets = sorted(kaggle_secrets, key=lambda s: not bool(s.get("is_default", False)))
            for s in sorted_secrets:
                u = s.get("username") or ""
                t = s.get("secret_value") or ""
                if u and t:
                    accounts.append({
                        "username": u,
                        "token": t,
                        "is_default": bool(s.get("is_default", False)),
                    })
            if accounts:
                return accounts

    # 2. Fallback: parse .kaggle_users dotfile
    target_file = find_kaggle_users_file(users_file)
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
                    u = m_user.group(1).strip()
                    accounts.append({
                        "username": u,
                        "token": m_token.group(1).strip(),
                        "is_default": u == "lemtreursi",
                    })
            accounts.sort(key=lambda a: not a.get("is_default", False))
        except OSError as e:
            logger.warning("Could not read .kaggle_users at %s: %s", target_file, e)

    return accounts


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
    6. Default fallback account (lemtreursi)
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

    # User home ~/.kaggle/kaggle.json fallback (supporting utf-8-sig, utf-16, utf-8)
    user_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not (k_user and k_key) and user_kaggle_json.exists():
        for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
            try:
                creds = json.loads(user_kaggle_json.read_text(encoding=enc))
                if not k_user:
                    k_user = creds.get("username", "")
                if not k_key:
                    k_key = creds.get("key", "")
                break
            except (OSError, ValueError, UnicodeError) as e:
                logger.debug("Failed parsing ~/.kaggle/kaggle.json with %s: %s", enc, e)

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

    if not k_user:
        k_user = "lemtreursi"

    # Export to environment for Kaggle Python SDK and child processes
    if k_user:
        os.environ["KAGGLE_USERNAME"] = k_user
    if k_key:
        os.environ["KAGGLE_KEY"] = k_key

    # Sync ~/.kaggle/kaggle.json in clean UTF-8 so CLI works seamlessly
    if k_user and k_key:
        try:
            k_json = Path.home() / ".kaggle" / "kaggle.json"
            k_json.parent.mkdir(parents=True, exist_ok=True)
            k_content = json.dumps({"username": k_user, "key": k_key}, indent=2)
            k_json.write_text(k_content, encoding="utf-8")
        except OSError:
            pass

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

