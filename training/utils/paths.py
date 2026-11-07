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


def get_models_hub_root(project_root: Path | None = None) -> Path:
    """Return the root directory of the ``LemGendaryModels`` hub repository.

    Honours ``KAGGLE_WORKING_DIR`` in cloud execution, an explicit isolated ``project_root``
    when tests pass one that differs from the canonical project root, and otherwise resolves
    ``get_workspace_root() / "LemGendaryModels"``.
    """
    if project_root is not None:
        root = Path(project_root).resolve()
        canon_root = get_project_root().resolve()
        if root != canon_root:
            if (root.parent / "LemGendaryModels").exists():
                return (root.parent / "LemGendaryModels").resolve()
            return (root / "LemGendaryModels").resolve()

    if os.environ.get("KAGGLE_WORKING_DIR") or "/kaggle/working" in str(Path.cwd()):
        return Path("/kaggle/working/LemGendaryModels").resolve()

    return (get_workspace_root() / "LemGendaryModels").resolve()


def get_model_hub_dir(
    model_key: str,
    project_root: Path | None = None,
    *,
    create: bool = True,
) -> Path:
    """Return the model directory ``LemGendaryModels/<model_key>``."""
    hub_dir = (get_models_hub_root(project_root) / model_key).resolve()
    if create:
        hub_dir.mkdir(parents=True, exist_ok=True)
    return hub_dir


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
