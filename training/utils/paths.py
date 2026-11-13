"""Path resolution and repository root auto-discovery for LemGendary Model Training Suite."""

import functools
import os
import sys
import sysconfig
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


MODELS_FOLDER = "LemGendaryModels"
"""Folder of the trained models hub inside the workspace root."""


def get_models_hub_root(project_root: Path | None = None) -> Path:
    """Return the root directory of the ``LemGendaryModels`` hub: the one hub-root resolver.

    The hub is ``get_workspace_root() / "LemGendaryModels"``: ``LEMGENDARY_WORKSPACE_ROOT``
    when set (the Kaggle notebook sets it to ``/kaggle/working`` while the suite runs from its
    ``/kaggle/temp`` clone, so checkpoints land in the kernel output), else the project root's
    parent (the local sibling checkout). An explicit ``project_root`` that differs from the
    canonical one (an isolated test tree) resolves next to that tree instead.
    """
    if project_root is not None:
        root = Path(project_root).resolve()
        canon_root = get_project_root().resolve()
        if root != canon_root:
            if (root.parent / MODELS_FOLDER).exists():
                return (root.parent / MODELS_FOLDER).resolve()
            return (root / MODELS_FOLDER).resolve()

    return (get_workspace_root() / MODELS_FOLDER).resolve()


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


def venv_site_packages(venv_dir: Path) -> Path:
    """``site-packages`` of the virtual environment at ``venv_dir`` on this platform.

    Windows venvs use ``Lib/site-packages``; POSIX venvs ``lib/pythonX.Y/site-packages``.
    The layout comes from the interpreter's ``venv`` install scheme.
    """
    base = str(venv_dir)
    purelib = sysconfig.get_path("purelib", scheme="venv", vars={"base": base, "platbase": base})
    return Path(os.path.normpath(purelib))


def bootstrap_sys_path() -> None:
    """Ensure project root and sibling dependency fallbacks are on sys.path."""
    proj_root = str(get_project_root())
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    venv_site_pkgs = str(venv_site_packages(Path(proj_root) / ".venv"))
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
