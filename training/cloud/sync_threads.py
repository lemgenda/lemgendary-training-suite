"""Background cloud-sync threads with a bounded, logged wait at interpreter exit (R2-T9).

Uploads used to run on non-daemon threads: Python's shutdown joins every non-daemon
thread without a limit, so a stalled Hugging Face/Kaggle/Drive upload kept a finished
(or interrupted) training process alive indefinitely, and nothing logged what was still
pending. A daemon thread alone has the opposite failure: the interpreter kills it at
exit and a half-written upload is lost silently.

The usual pattern sits between the two: the upload threads are daemon threads, and an
``atexit`` handler (which runs before daemon threads are torn down) joins the pending
ones with one shared deadline and logs, per thread, whether it finished or was
abandoned. The deadline is ``fleet.sync_exit_timeout_sec`` (default
:data:`DEFAULT_EXIT_TIMEOUT_SEC`, 300 s: long enough for a checkpoint push on a normal
link, short enough to fit inside the grace period of a Kaggle/Colab session).
"""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger("lemtrain.cloud_sync")

DEFAULT_EXIT_TIMEOUT_SEC = 300.0
"""Seconds the process waits at exit for pending uploads when the config sets none."""

CONFIG_KEY = "sync_exit_timeout_sec"
"""Key under ``fleet`` overriding :data:`DEFAULT_EXIT_TIMEOUT_SEC`."""


def exit_timeout(config: Mapping[str, Any] | None) -> float:
    """The configured exit wait in seconds (``fleet.sync_exit_timeout_sec``, >= 0).

    Raises:
        ValueError: the configured value is negative or not a number.
    """
    fleet = (config or {}).get("fleet") or {}
    raw = fleet.get(CONFIG_KEY, DEFAULT_EXIT_TIMEOUT_SEC) if isinstance(fleet, Mapping) else None
    if raw is None:
        return DEFAULT_EXIT_TIMEOUT_SEC
    value = float(raw)
    if math.isnan(value) or value < 0.0:
        raise ValueError(f"fleet.{CONFIG_KEY} must be >= 0, got {raw!r}")
    return value


class SyncThreadRegistry:
    """Starts upload threads and waits for the pending ones once, at exit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._timeout = DEFAULT_EXIT_TIMEOUT_SEC

    def start(self, target: Callable[[], object], name: str, timeout: float) -> threading.Thread:
        """Run ``target`` on a daemon thread that the exit wait will cover."""
        thread = threading.Thread(target=target, name=name, daemon=True)
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
            self._timeout = timeout
        thread.start()
        return thread

    def pending(self) -> list[threading.Thread]:
        """Upload threads still running."""
        with self._lock:
            return [t for t in self._threads if t.is_alive()]

    def join_pending(self, timeout: float | None = None) -> tuple[int, int]:
        """Wait up to ``timeout`` seconds in total; return ``(finished, abandoned)``."""
        threads = self.pending()
        if not threads:
            return 0, 0
        budget = self._timeout if timeout is None else timeout
        logger.info(
            "Waiting up to %.0f s for %d pending cloud upload(s) before exit.",
            budget,
            len(threads),
        )
        deadline = time.monotonic() + budget
        abandoned: list[str] = []
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                abandoned.append(thread.name)
        if abandoned:
            logger.warning(
                "Exit wait of %.0f s elapsed; abandoning unfinished upload(s): %s. "
                "Run the cloud sync again to complete them.",
                budget,
                ", ".join(abandoned),
            )
        else:
            logger.info("All %d pending cloud upload(s) finished before exit.", len(threads))
        return len(threads) - len(abandoned), len(abandoned)


SYNC_THREADS = SyncThreadRegistry()
atexit.register(SYNC_THREADS.join_pending)
