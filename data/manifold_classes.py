"""The class names a manifold declares: the ``classes`` list of its ``dataset_info.yaml``.

The dataset compiler records the class names of labelled manifolds there (id order); a
JSON label that names a class (``class_name``) maps to its id through this list.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("lemtrain.data.manifold_classes")

INFO_FILE = "dataset_info.yaml"


class ManifoldClassesError(ValueError):
    """``dataset_info.yaml`` ``classes`` is present but not a list of non-empty names."""


def manifold_classes(manifold_dir: Path | str) -> tuple[str, ...]:
    """Class names in id order from ``<manifold_dir>/dataset_info.yaml`` (``()`` when none).

    A missing file or a file without ``classes`` gives ``()``. An unreadable file is logged
    and gives ``()``; a ``classes`` value that is not a list of non-empty strings raises
    :class:`ManifoldClassesError`.
    """
    info_file = Path(manifold_dir) / INFO_FILE
    if not info_file.is_file():
        return ()
    try:
        info = yaml.safe_load(info_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.warning("Could not read classes from %s: %s", info_file, exc)
        return ()
    classes = info.get("classes") if isinstance(info, dict) else None
    if classes is None:
        return ()
    if not (
        isinstance(classes, list)
        and all(isinstance(name, str) and name.strip() for name in classes)
    ):
        raise ManifoldClassesError(f"{info_file}: classes is not a list of non-empty names")
    return tuple(name.strip() for name in classes)
