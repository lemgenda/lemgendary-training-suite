"""Structured credentials and secrets loader for LemGendary Model Training Suite."""

import logging
import os
from pathlib import Path
from training.utils.paths import get_project_root, get_workspace_root

logger = logging.getLogger("lemtrain.secrets")

_SECRET_MAPPING = [
    ("GITHUB_PAT", ".GITHUB_PAT"),
    ("SUITE_PAT", ".SUITE_PAT"),
    ("SATURN_PAT", ".SATURN_PAT"),
    ("GOOGLE_DRIVE", ".GOOGLE_DRIVE"),
    ("KAGGLE_KEY", ".kaggle_token"),
    ("KAGGLE_USERNAME", ".kaggle_users"),
]


def _mask_secret(val: str) -> str:
    """Mask sensitive string leaving only leading and trailing characters."""
    if not val:
        return ""
    if len(val) <= 6:
        return "***"
    return f"{val[:3]}...{val[-3:]}"


def load_secrets(project_root: Path | None = None) -> dict[str, str]:
    """
    Mount local dotfile secrets into os.environ if not already present.
    Returns masked summary dictionary safe for diagnostic reporting.
    """
    proj_root = project_root or get_project_root()
    workspace_root = get_workspace_root()

    summary: dict[str, str] = {}

    for env_var, filename in _SECRET_MAPPING:
        # Check if already present in environment
        existing = os.environ.get(env_var, "").strip()
        if existing:
            summary[env_var] = _mask_secret(existing)
            continue

        # Look in project root then workspace root
        candidate_paths = [
            proj_root / filename,
            workspace_root / filename,
        ]

        found_val = ""
        for path in candidate_paths:
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        found_val = content
                        break
                except OSError as exc:
                    logger.warning("Failed to read secret file %s: %s", path, exc)

        if found_val:
            os.environ[env_var] = found_val
            summary[env_var] = _mask_secret(found_val)
        else:
            summary[env_var] = "[NOT_SET]"

    return summary


def get_secret(name: str, default: str | None = None) -> str | None:
    """Retrieve secret from environment, attempting automated dotfile load if missing."""
    val = os.environ.get(name)
    if val:
        return val.strip()

    # Attempt load
    load_secrets()
    val = os.environ.get(name)
    if val:
        return val.strip()

    return default
