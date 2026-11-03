"""Interrupt handling and child process cleanup for LemGendary Model Training Suite."""

import atexit
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("lemtrain.interrupt")

_ACTIVE_PROCESSES: list[subprocess.Popen[Any]] = []


@dataclass
class _InterruptState:
    """Process-wide interrupt bookkeeping shared by the signal handler and its installers."""

    emergency_sync_handler: Callable[[], None] | None = None
    handlers_installed: bool = False
    exiting: bool = False


_STATE = _InterruptState()


def register_active_process(proc: subprocess.Popen[Any]) -> None:
    """Track an active subprocess for guaranteed termination upon exit."""
    if proc not in _ACTIVE_PROCESSES:
        _ACTIVE_PROCESSES.append(proc)


def cleanup_active_processes() -> None:
    """Terminate all tracked active child processes across Windows and POSIX."""
    if not _ACTIVE_PROCESSES:
        return

    logger.info("Terminating %d active child processes...", len(_ACTIVE_PROCESSES))
    for p in _ACTIVE_PROCESSES:
        if p.poll() is None:
            try:
                if os.name == "nt":
                    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        capture_output=True,
                        check=False,
                        creationflags=cflags,
                    )
                else:
                    p.terminate()
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning(
                    "Failed to terminate subprocess PID %s: %s", getattr(p, "pid", "unknown"), exc
                )
    _ACTIVE_PROCESSES.clear()


atexit.register(cleanup_active_processes)


def register_emergency_sync(handler: Callable[[], None]) -> None:
    """Register callback to be invoked during preemption or SIGINT / SIGTERM interruption."""
    _STATE.emergency_sync_handler = handler


def graceful_exit(signum: int, _frame: Any) -> None:
    """Signal handler executing emergency sync before termination.

    The ``(signum, frame)`` signature is fixed by :func:`signal.signal`.
    """
    if _STATE.exiting:
        return
    _STATE.exiting = True

    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.warning("Signal %s detected! Initiating graceful exit protocol...", sig_name)

    # Only rank 0 / MainProcess runs emergency remote sync
    is_main = multiprocessing.current_process().name == "MainProcess"
    rank_env = os.environ.get("RANK", "0")
    handler = _STATE.emergency_sync_handler
    try:
        if is_main and rank_env in ("0", "") and handler is not None:
            logger.info("Executing registered emergency checkpoint sync...")
            try:
                handler()
            except Exception:
                # The callback is arbitrary user code: record the traceback, then let the
                # finally block below still clean up children and exit with the signal code.
                logger.exception("Emergency sync callback failed")
                raise
    finally:
        cleanup_active_processes()
        sys.exit(128 + signum)


def silent_worker_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any
) -> None:
    """Excepthook for multiprocessing workers: shutdown errors get one line, not a traceback.

    Interrupts and closed pipes are expected when a worker is stopped; they are still logged.
    """
    if issubclass(exc_type, (KeyboardInterrupt, EOFError, BrokenPipeError, ConnectionResetError)):
        logger.info("Worker stopped: %s: %s", exc_type.__name__, exc_value)
        return
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def install_signal_handlers() -> None:
    """Install SIGINT and SIGTERM handlers once in main process."""
    if _STATE.handlers_installed:
        return

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)
    _STATE.handlers_installed = True
