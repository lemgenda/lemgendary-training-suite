"""Path resolution and repository root auto-discovery for LemGendary Model Training Suite."""

import functools
import os
import sys
from pathlib import Path


@functools.cache
def get_project_root() -> Path:
    """Discover the canonical root directory of lemgendary-training-suite (cached per process)."""
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "unified_models_v2.yaml").exists() or (candidate / "config.yaml").exists():
        return candidate

    cwd = Path.cwd().resolve()
    if (cwd / "unified_models_v2.yaml").exists() or (cwd / "config.yaml").exists():
        return cwd

    # Fallback to parent candidate
    return candidate


WORKSPACE_ROOT_ENV = "LEMGENDARY_WORKSPACE_ROOT"
"""Environment variable that overrides the workspace root."""


def get_workspace_root() -> Path:
    """Discover the workspace root containing sibling LemGendary projects.

    ``LEMGENDARY_WORKSPACE_ROOT`` wins when set (read on every call so tests and launchers can
    switch it); otherwise this is the parent of the project root, where sibling projects such as
    lemgendary-datasets and lemgendary-env-manager live.
    """
    override = os.environ.get(WORKSPACE_ROOT_ENV)
    if override:
        return Path(override).resolve()
    return get_project_root().parent


def resolve_path(target: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    path = Path(target)
    if path.is_absolute():
        return path
    return (get_project_root() / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return resolved Path."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def bootstrap_sys_path() -> None:
    """Ensure project root and sibling dependency fallbacks are on sys.path."""
    proj_root = str(get_project_root())
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    venv_site_pkgs = os.path.normpath(os.path.join(proj_root, ".venv", "Lib", "site-packages"))
    if os.path.exists(venv_site_pkgs) and venv_site_pkgs not in sys.path:
        sys.path.insert(1, venv_site_pkgs)


def unique_resolved(paths: list[Path]) -> list[Path]:
    """Resolve ``paths`` and drop duplicates, keeping the first occurrence's order."""
    unique: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique
