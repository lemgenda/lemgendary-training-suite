"""Structured credentials and secrets loader for LemGendary Model Training Suite."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from training.utils.paths import get_project_root, get_workspace_root

logger = logging.getLogger("lemtrain.secrets")

SECRETS_FILE_NAME = ".secrets.yaml"

_SERVICE_ENV_MAPPINGS: list[tuple[str, str, str | None]] = [
    ("GITHUB_PAT", "github", None),
    ("SUITE_PAT", "github", None),
    ("SATURN_PAT", "saturn_cloud", None),
    ("GOOGLE_DRIVE", "google_drive", None),
    ("GOOGLE_DRIVE_JSON", "google_drive", None),
    ("KAGGLE_KEY", "kaggle", None),
    ("KAGGLE_USERNAME", "kaggle", None),
    ("DEEPSEEK_API_KEY", "deepseek", None),
]


def _mask_secret(val: str) -> str:
    """Mask sensitive string leaving only leading and trailing characters."""
    if not val:
        return ""
    if len(val) <= 6:
        return "***"
    return f"{val[:3]}...{val[-3:]}"


def find_secrets_yaml(project_root: Path | None = None) -> Path | None:
    """Locate the workspace root .secrets.yaml file."""
    proj = project_root or get_project_root()
    workspace = get_workspace_root()

    candidates = [
        workspace / SECRETS_FILE_NAME,
        proj.parent / SECRETS_FILE_NAME,
        proj / SECRETS_FILE_NAME,
        Path.cwd() / SECRETS_FILE_NAME,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_secrets_records(project_root: Path | None = None) -> list[dict[str, Any]]:
    """Return all secret records parsed from .secrets.yaml."""
    path = find_secrets_yaml(project_root)
    if not path:
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as exc:
        logger.warning("Failed parsing %s: %s", path, exc)
        return []
    records = data.get("secrets", []) if isinstance(data, dict) else []
    return [r for r in records if isinstance(r, dict)]


def load_secrets(project_root: Path | None = None) -> dict[str, str]:
    """Mount secrets from .secrets.yaml into os.environ if not already present.

    Returns masked summary dictionary safe for diagnostic reporting.
    """
    summary: dict[str, str] = {}
    records = load_secrets_records(project_root)

    # Group by service
    by_service: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        svc = str(r.get("service", "")).lower()
        by_service.setdefault(svc, []).append(r)

    for env_var, service_name, _ in _SERVICE_ENV_MAPPINGS:
        existing = os.environ.get(env_var, "").strip()
        if existing:
            summary[env_var] = _mask_secret(existing)
            continue

        svc_records = by_service.get(service_name, [])
        found_val = ""

        if svc_records:
            default_rec = next(
                (r for r in svc_records if bool(r.get("is_default", False))),
                svc_records[0],
            )
            if env_var == "KAGGLE_USERNAME":
                found_val = str(default_rec.get("username") or "").strip()
            else:
                found_val = str(default_rec.get("secret_value") or "").strip()

        if found_val:
            os.environ[env_var] = found_val
            summary[env_var] = _mask_secret(found_val)
        else:
            summary[env_var] = "[NOT_SET]"

    return summary


def get_secret(name: str, default: str | None = None) -> str | None:
    """Retrieve secret from environment, attempting automated .secrets.yaml load if missing."""
    val = os.environ.get(name)
    if val:
        return val.strip()

    load_secrets()
    val = os.environ.get(name)
    if val:
        return val.strip()

    return default
