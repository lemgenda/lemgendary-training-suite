"""Interrupt handling and child process cleanup for LemGendary Model Training Suite."""

import atexit
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("lemtrain.interrupt")

_ACTIVE_PROCESSES: list[subprocess.Popen[Any]] = []
_EMERGENCY_SYNC_HANDLER: Callable[[], None] | None = None
_HANDLERS_INSTALLED: bool = False


def register_active_process(proc: subprocess.Popen[Any]) -> None:
    """Track an active subprocess for guaranteed termination upon exit."""
    if proc not in _ACTIVE_PROCESSES:
        _ACTIVE_PROCESSES.append(proc)


def cleanup_active_processes(*args: Any) -> None:
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
                logger.warning("Failed to terminate subprocess PID %s: %s", getattr(p, "pid", "unknown"), exc)
    _ACTIVE_PROCESSES.clear()


atexit.register(cleanup_active_processes)


def register_emergency_sync(handler: Callable[[], None]) -> None:
    """Register callback to be invoked during preemption or SIGINT / SIGTERM interruption."""
    global _EMERGENCY_SYNC_HANDLER
    _EMERGENCY_SYNC_HANDLER = handler


_EXITING: bool = False

def graceful_exit(signum: int, frame: Any) -> None:
    """Signal handler executing emergency sync before termination."""
    global _EXITING
    if _EXITING:
        return
    _EXITING = True

    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.warning("Signal %s detected! Initiating graceful exit protocol...", sig_name)

    # Only rank 0 / MainProcess runs emergency remote sync
    import multiprocessing
    is_main = multiprocessing.current_process().name == "MainProcess"
    rank_env = os.environ.get("RANK", "0")
    if is_main and rank_env in ("0", "") and _EMERGENCY_SYNC_HANDLER is not None:
        try:
            logger.info("Executing registered emergency checkpoint sync...")
            _EMERGENCY_SYNC_HANDLER()
        except Exception:
            # Signal-handler boundary: the registered callback is arbitrary user code and any
            # failure must be reported without aborting child-process cleanup and exit below.
            logger.exception("Emergency sync callback failed")

    cleanup_active_processes()
    sys.exit(128 + signum)


def silent_worker_excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None:
    """Quiet excepthook for multiprocessing workers to prevent SIGINT trace dump spam."""
    if issubclass(exc_type, (KeyboardInterrupt, EOFError, BrokenPipeError, ConnectionResetError)):
        return
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def install_signal_handlers() -> None:
    """Install SIGINT and SIGTERM handlers once in main process."""
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)
    _HANDLERS_INSTALLED = True
