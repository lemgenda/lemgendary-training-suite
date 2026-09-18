"""Backward-compatibility adapter delegating to modular training.checkpoint subsystem."""

import logging
import os
from pathlib import Path
import time
from typing import Any

from training.checkpoint.manager import safe_atomic_save
from training.checkpoint.resume import stretch_scheduler_runway

logger = logging.getLogger("lemtrain.sota_rollback")


def safe_torch_save(obj: Any, path: str | Path) -> bool:
    """Saves a torch object with disk-space auditing and atomic replacement."""
    return safe_atomic_save(obj, path)


def load_scheduler_state_stretched(
    scheduler: Any,
    state_dict: dict[str, Any],
    current_total_steps: int,
    expected_step: int | None = None,
) -> None:
    """Loads scheduler state dict while stretching the runway if total_steps mismatch."""
    stretch_scheduler_runway(scheduler, state_dict, current_total_steps, expected_step=expected_step)


def safe_replace(src: str | Path, dst: str | Path) -> bool:
    """Atomic file replacement with Windows file lock defense."""
    src_p = Path(src)
    dst_p = Path(dst)
    max_retries = 15
    base_delay = 0.5

    for i in range(max_retries):
        try:
            if dst_p.exists():
                temp_old = dst_p.with_name(f"{dst_p.name}.old_{int(time.time())}")
                dst_p.rename(temp_old)
                src_p.rename(dst_p)
                try:
                    temp_old.unlink()
                except OSError as exc:
                    logger.debug("Deferred temporary old file removal for %s: %s", temp_old, exc)
            else:
                src_p.rename(dst_p)
            return True
        except (PermissionError, OSError) as exc:
            logger.debug("File replacement retry %d/%d for %s: %s", i + 1, max_retries, dst_p, exc)
            time.sleep(base_delay * (1.5 ** i))
    return False
