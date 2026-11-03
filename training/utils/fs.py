"""Filesystem helpers shared across the suite.

``remove_tree`` replaces ``shutil.rmtree(..., ignore_errors=True)``: a directory
that cannot be removed is reported in the log and to the caller instead of
being silently left behind.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("lemtrain.utils.fs")


def remove_tree(path: Path, *, purpose: str) -> bool:
    """Recursively delete ``path``.

    Args:
        path: Directory to remove.
        purpose: Short description of why the directory is removed, used in
            the log message when removal fails.

    Returns:
        True when the directory is gone afterwards (removed, or already
        absent), False when an ``OSError`` prevented removal.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        logger.debug("Directory %s already absent (%s)", path, purpose)
    except OSError as exc:
        logger.warning("Could not remove directory %s (%s): %s", path, purpose, exc)
        return False
    return True
