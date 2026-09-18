"""Path resolution and repository root auto-discovery for LemGendary Model Training Suite."""

import os
import sys
from pathlib import Path

_CACHED_PROJECT_ROOT: Path | None = None
_CACHED_WORKSPACE_ROOT: Path | None = None


def get_project_root() -> Path:
    """Discover the canonical root directory of lemgendary-training-suite."""
    global _CACHED_PROJECT_ROOT
    if _CACHED_PROJECT_ROOT is not None:
        return _CACHED_PROJECT_ROOT

    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "unified_models_v2.yaml").exists() or (candidate / "config.yaml").exists():
        _CACHED_PROJECT_ROOT = candidate
        return _CACHED_PROJECT_ROOT

    cwd = Path.cwd().resolve()
    if (cwd / "unified_models_v2.yaml").exists() or (cwd / "config.yaml").exists():
        _CACHED_PROJECT_ROOT = cwd
        return _CACHED_PROJECT_ROOT

    # Fallback to parent candidate
    _CACHED_PROJECT_ROOT = candidate
    return _CACHED_PROJECT_ROOT


def get_workspace_root() -> Path:
    """Discover the workspace root containing sibling LemGendary projects."""
    global _CACHED_WORKSPACE_ROOT
    if _CACHED_WORKSPACE_ROOT is not None:
        return _CACHED_WORKSPACE_ROOT

    proj_root = get_project_root()
    parent = proj_root.parent
    if (parent / "lemgendary-datasets").exists() or (parent / "lemgendary-env-manager").exists():
        _CACHED_WORKSPACE_ROOT = parent
        return _CACHED_WORKSPACE_ROOT

    _CACHED_WORKSPACE_ROOT = parent
    return _CACHED_WORKSPACE_ROOT


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
