"""Process-wide stdout/stderr tee that feeds job log lines to the JobManager."""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("lemtrain.server.jobs")


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

CLEAN_CONTROL_RE = re.compile(r"^\s*\[K\s*")

PROGRESS_RE = re.compile(
    r"(\d+%\s*[━█─\-=|]+|\b\d+/\d+\b\s+.*\b(it/s|s/it|B/s|KB/s|MB/s|GB/s)\b|\d+:\d+<\d+:\d+|\d+"
    r"%\s*\|.*\|\s*\d+/\d+|\[BATCH\s+\d+/\d+\])"
)


def _clean_console_line(text: str) -> str:
    if not text:
        return ""
    if "\r" in text:
        parts = [p for p in text.split("\r") if p.strip()]
        if parts:
            text = parts[-1]
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CLEAN_CONTROL_RE.sub("", text)
    return text.strip()


def _is_progress_line(text: str) -> bool:
    return bool(PROGRESS_RE.search(text))


# Failures a tee line handler (JobManager._broadcast_log) can raise: job-state DB access
# (sqlite3), log-file I/O (OSError), payload encoding (TypeError/ValueError) and a closed
# event loop (RuntimeError).
_LINE_HANDLER_ERRORS: tuple[type[Exception], ...] = (
    sqlite3.Error,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

# Failures writing to the wrapped console stream: broken pipe / detached console (OSError) and
# closed stream or unencodable text (ValueError, incl. UnicodeEncodeError).
_STREAM_ERRORS: tuple[type[Exception], ...] = (OSError, ValueError)


class _GlobalOutputTee:
    """Process-wide stream tee routing thread-specific stdout/stderr lines to job loggers."""

    def __init__(self, original_stream: Any) -> None:
        self.original_stream = original_stream
        self._handlers: dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()
        self._buffers: dict[int, str] = {}
        self._diag_state = threading.local()

    def _report(self, msg: str, exc: Exception) -> None:
        """Log a tee failure without re-entering the tee recursively.

        Must be called with ``self._lock`` released: the log record may be written back
        through this tee (sys.stdout/sys.stderr), which would otherwise deadlock.
        """
        if getattr(self._diag_state, "active", False):
            return
        self._diag_state.active = True
        try:
            logger.warning("%s: %s", msg, exc)
        finally:
            self._diag_state.active = False

    def _dispatch(
        self, handler: Callable[[str], None], line: str, failures: list[Exception]
    ) -> None:
        try:
            handler(line)
        except _LINE_HANDLER_ERRORS as exc:
            failures.append(exc)

    def _report_failures(self, failures: list[Exception]) -> None:
        for exc in failures:
            self._report("Job log line handler failed; line dropped from job log", exc)

    def _write_original(self, s: str | None) -> None:
        try:
            if s is not None:
                self.original_stream.write(s)
            self.original_stream.flush()
        except _STREAM_ERRORS as exc:
            self._report("Console stream write failed", exc)

    def register_thread(self, thread_id: int, on_line: Callable[[str], None]) -> None:
        """Route complete lines written by ``thread_id`` to ``on_line``."""
        with self._lock:
            self._handlers[thread_id] = on_line
            self._buffers[thread_id] = ""

    def unregister_thread(self, thread_id: int) -> None:
        """Flush any buffered partial line for ``thread_id`` and stop routing it."""
        failures: list[Exception] = []
        with self._lock:
            buf = self._buffers.pop(thread_id, "").strip()
            handler = self._handlers.pop(thread_id, None)
            if buf and handler:
                self._dispatch(handler, buf, failures)
        self._report_failures(failures)

    def write(self, s: str) -> int:
        """Write to the original stream and dispatch complete lines of the calling thread."""
        self._write_original(s)

        tid = threading.get_ident()
        failures: list[Exception] = []
        with self._lock:
            handler = self._handlers.get(tid)
            if handler:
                buf = self._buffers.get(tid, "") + s
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    clean = _clean_console_line(line)
                    if clean:
                        self._dispatch(handler, clean, failures)
                self._buffers[tid] = buf
        self._report_failures(failures)
        return len(s)

    def flush(self) -> None:
        """Flush the original stream and dispatch the calling thread's partial line."""
        self._write_original(None)
        tid = threading.get_ident()
        failures: list[Exception] = []
        with self._lock:
            handler = self._handlers.get(tid)
            if handler:
                buf = self._buffers.get(tid, "")
                clean = _clean_console_line(buf)
                if clean:
                    self._dispatch(handler, clean, failures)
                    self._buffers[tid] = ""
        self._report_failures(failures)

    def isatty(self) -> bool:
        """Report a non-interactive stream so libraries disable TTY-only output."""
        return False

    def fileno(self) -> int:
        """Return the wrapped stream's file descriptor."""
        if hasattr(self.original_stream, "fileno"):
            return self.original_stream.fileno()
        raise OSError("fileno not supported")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_stream, name)


# Process-wide tees, installed once: {"stdout": tee, "stderr": tee}.
_INSTALLED_TEES: dict[str, _GlobalOutputTee] = {}


def _ensure_output_tees() -> tuple[_GlobalOutputTee, _GlobalOutputTee]:
    """Install the stdout/stderr tees on first use and return them."""
    for name in ("stdout", "stderr"):
        if name not in _INSTALLED_TEES:
            tee = _GlobalOutputTee(getattr(sys, name))
            _INSTALLED_TEES[name] = tee
            setattr(sys, name, tee)
    return _INSTALLED_TEES["stdout"], _INSTALLED_TEES["stderr"]
