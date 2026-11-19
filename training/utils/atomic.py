"""Atomic file system operations with exponential backoff on locks."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("lemtrain.utils.atomic")


def remove_temp_file_safely(tmp_path: Path) -> None:
    """Clean up a temporary file, suppressing and logging any OS error."""
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError as err:
            logger.debug("Failed cleaning temporary file %s: %s", tmp_path, err)
