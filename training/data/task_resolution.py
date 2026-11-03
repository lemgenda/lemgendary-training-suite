"""Resolve a manifold's task: ``dataset_info.yaml`` first, the datasets taxonomy for legacy ones.

Order:

1. ``task_type`` in the manifold's ``dataset_info.yaml`` (the compiler writes the plain
   datasets task id, e.g. ``pose``). Older manifolds may carry a training-suite name
   (e.g. ``face_detection``); both are accepted.
2. Legacy manifolds without ``task_type``: the folder layout the old heuristic used
   (forex markers, then ``masks/``, ``labels/``, ``targets/``) names a datasets task id
   (:data:`LEGACY_LAYOUT_TASKS`), which must exist in the datasets
   ``task_taxonomy.json`` with that split folder as its ``layout_dir``.
3. Otherwise :class:`TaskResolutionError`; there is no silent default.

:data:`DATASETS_TO_INTERNAL_TASK` is the one table that turns a datasets task id into
the training suite's internal task name.

The taxonomy file is ``LEMGENDARY_TASK_TAXONOMY_FILE`` when set, else
``<workspace_root>/lemgendary-datasets/task_taxonomy.json`` (the workspace root of
:func:`training.utils.paths.get_workspace_root`, as for ``training.ecosystem``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from training.utils.paths import WORKSPACE_ROOT_ENV, get_workspace_root

TASK_TAXONOMY_ENV = "LEMGENDARY_TASK_TAXONOMY_FILE"
"""Environment variable that overrides the taxonomy file location."""

DATASETS_FOLDER = "lemgendary-datasets"
"""Folder that owns ``task_taxonomy.json``."""

DATASETS_TO_INTERNAL_TASK: Mapping[str, str] = MappingProxyType(
    {
        "restoration": "restoration",
        "super-resolution": "restoration",
        "segmentation": "segmentation",
        "pose": "face_detection",
        "detection": "detection",
        "yolo": "yolo",
        "quality": "quality",
        "authenticity": "quality",
        "classification": "classification",
        "parameter_prediction": "parameter_prediction",
        "forex": "forex",
        "ai_helper": "text_instruction",
    }
)
"""Datasets task id (taxonomy ``tasks[].id`` or documentation task) -> internal task name."""

INTERNAL_TASKS: frozenset[str] = frozenset(DATASETS_TO_INTERNAL_TASK.values()) | {
    "face",
    "enhancement",
}
"""Internal task names a ``dataset_info.yaml`` may already carry (kept as they are)."""

FOREX_TASK = "forex"
"""Datasets task id of forex manifolds (a documentation task in the taxonomy)."""

LEGACY_LAYOUT_TASKS: tuple[tuple[str, str], ...] = (
    ("masks", "segmentation"),
    ("labels", "pose"),
    ("targets", "restoration"),
)
"""Split folder of a legacy manifold -> datasets task id, first present folder wins."""


class TaskResolutionError(RuntimeError):
    """The manifold's task cannot be determined, or the taxonomy file is unusable."""


@dataclass(frozen=True)
class TaskTaxonomy:
    """The parts of the datasets ``task_taxonomy.json`` the training suite uses."""

    path: Path
    layout_dirs: Mapping[str, str | None]
    documentation_tasks: frozenset[str]

    def owns_layout(self, task_id: str, folder: str) -> bool:
        """True when ``task_id`` is a taxonomy task whose split folder is ``folder``."""
        return task_id in self.layout_dirs and self.layout_dirs[task_id] == folder


def taxonomy_file() -> Path:
    """Path of the taxonomy file that :func:`task_taxonomy` reads."""
    override = os.environ.get(TASK_TAXONOMY_ENV)
    if override:
        return Path(override)
    return get_workspace_root() / DATASETS_FOLDER / "task_taxonomy.json"


def _invalid(path: Path, message: str) -> TaskResolutionError:
    return TaskResolutionError(f"Invalid task taxonomy {path}: {message}")


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskResolutionError(
            f"Task taxonomy not found: expected {path} (check out {DATASETS_FOLDER} next to "
            f"this repository, or set {TASK_TAXONOMY_ENV} / {WORKSPACE_ROOT_ENV})"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskResolutionError(f"Cannot read task taxonomy {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise _invalid(path, "top level is not an object")
    return raw


@cache
def _load(path: Path) -> TaskTaxonomy:
    raw = _read(path)
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise _invalid(path, "'tasks' must be a non-empty list")
    layout_dirs: dict[str, str | None] = {}
    for entry in tasks:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise _invalid(path, "every task needs a string 'id'")
        layout = entry.get("layout_dir")
        if layout is not None and not isinstance(layout, str):
            raise _invalid(path, f"task '{entry['id']}': layout_dir must be a string or null")
        layout_dirs[entry["id"]] = layout
    doc_tasks = raw.get("documentation_tasks", {})
    if not isinstance(doc_tasks, dict):
        raise _invalid(path, "'documentation_tasks' must be an object")
    return TaskTaxonomy(
        path=path,
        layout_dirs=MappingProxyType(layout_dirs),
        documentation_tasks=frozenset(str(v) for v in doc_tasks.values()),
    )


def task_taxonomy() -> TaskTaxonomy:
    """Return the parsed taxonomy file, cached per path."""
    return _load(taxonomy_file())


def clear_cache() -> None:
    """Forget parsed files (tests that switch ``LEMGENDARY_TASK_TAXONOMY_FILE``)."""
    _load.cache_clear()


def to_internal_task(task_id: str, source: str) -> str:
    """Internal task name of a datasets task id (or of an internal name, unchanged)."""
    if task_id in DATASETS_TO_INTERNAL_TASK:
        return DATASETS_TO_INTERNAL_TASK[task_id]
    if task_id in INTERNAL_TASKS:
        return task_id
    raise TaskResolutionError(
        f"Task '{task_id}' from {source} has no training-suite equivalent; known datasets "
        f"task ids: {sorted(DATASETS_TO_INTERNAL_TASK)}"
    )


@dataclass(frozen=True)
class ManifoldLayout:
    """The facts about a manifold folder that task resolution looks at."""

    path: Path
    name: str
    container_type: str
    metadata: Mapping[str, Any]

    def is_forex(self) -> bool:
        """Forex markers of the old heuristic: parquet container, name or ``forex/`` folder."""
        return (
            self.container_type == "parquet"
            or "forex" in self.name.lower()
            or (self.path / "forex").exists()
        )


def _legacy_task_id(layout: ManifoldLayout, taxonomy: TaskTaxonomy) -> str:
    if layout.is_forex():
        if FOREX_TASK not in taxonomy.documentation_tasks:
            raise _invalid(taxonomy.path, f"no '{FOREX_TASK}' documentation task")
        return FOREX_TASK
    for folder, task_id in LEGACY_LAYOUT_TASKS:
        if (layout.path / folder).exists():
            if not taxonomy.owns_layout(task_id, folder):
                raise _invalid(
                    taxonomy.path, f"legacy layout '{folder}/' expects task '{task_id}' with it"
                )
            return task_id
    raise TaskResolutionError(
        f"Cannot determine the task of manifold {layout.path}: dataset_info.yaml has no "
        f"task_type and the folder has no forex marker, masks/, labels/ or targets/. "
        f"Add 'task_type: <task id>' to {layout.path / 'dataset_info.yaml'}."
    )


def resolve_manifold_task(layout: ManifoldLayout) -> str:
    """Internal task name of a manifold (see the module docstring for the order)."""
    declared = layout.metadata.get("task_type")
    if isinstance(declared, list) and declared:
        declared = declared[0]
    if isinstance(declared, str) and declared.strip():
        return to_internal_task(declared.strip(), f"{layout.path} task_type")
    taxonomy = task_taxonomy()
    task_id = _legacy_task_id(layout, taxonomy)
    return to_internal_task(task_id, f"taxonomy {taxonomy.path}")
