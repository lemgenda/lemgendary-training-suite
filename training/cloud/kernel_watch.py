"""Watch a Kaggle kernel session until it ends: status with backoff, new log lines, pulls.

One loop for the GUI job (:mod:`training.server.job_kaggle_run`) and the terminal monitor
(:mod:`training.kaggle_stream`); they differ only in where lines go and how cancellation is
signalled, which they pass in as callbacks.

Each poll makes one ``kernels_status`` call and, while the kernel runs, one ``kernels_logs``
call; the interval backs off while the status is unchanged
(:class:`training.cloud.kernel_status.PollBackoff`). The optional ``on_poll`` hook runs on
every poll of a running kernel (the watchers use it to pull ``pytorch/default`` when the
Hub has a newer version, instead of re-pulling every variation on each epoch log line).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kernel_status import (
    RUNNING,
    TERMINAL_STATUSES,
    PollBackoff,
    failure_message,
    status_name,
)

logger = logging.getLogger("lemtrain.cloud.kernel_watch")


def _join_stream_items(items: list[Any], keep_plain: bool) -> str:
    """Join the ``data`` of stream objects; plain items are kept only when ``keep_plain``."""
    chunks = []
    for item in items:
        if isinstance(item, dict) and "data" in item:
            chunks.append(str(item["data"]))
        elif keep_plain:
            chunks.append(str(item))
    return "".join(chunks)


def _parse_text_logs(text: str) -> str:
    """Text of a string log response: a JSON stream array, one stream object or plain text."""
    try:
        entries = json.loads(text)
    except (ValueError, TypeError):
        return text
    if isinstance(entries, list):
        return _join_stream_items(entries, keep_plain=False)
    if isinstance(entries, dict) and "data" in entries:
        return str(entries["data"])
    return text


def parse_kernel_logs(raw_logs: Any) -> str:
    """Text of a ``kernels_logs`` response (JSON array of stream objects or plain text)."""
    if not raw_logs:
        return ""
    if isinstance(raw_logs, str):
        return _parse_text_logs(raw_logs)
    if isinstance(raw_logs, list):
        return _join_stream_items(raw_logs, keep_plain=True)
    return str(raw_logs)


@dataclass(frozen=True)
class WatchHooks:
    """Where the watcher sends lines, how it waits and how it learns about cancellation."""

    emit: Callable[[str], None]
    cancelled: Callable[[], bool] = lambda: False
    sleep: Callable[[float], object] = time.sleep
    on_poll: Callable[[], None] | None = None


@dataclass
class WatchResult:
    """How the watch ended: the last status (None when the caller cancelled the watch)."""

    status: str | None
    failure: str = ""
    lines: int = 0
    polls: int = field(default=0)


class KernelWatcher:
    """Poll one kernel until it reaches a terminal status or the caller cancels."""

    def __init__(self, api: Any, handle: str, backoff: PollBackoff, hooks: WatchHooks) -> None:
        self._api = api
        self._handle = handle
        self._backoff = backoff
        self._hooks = hooks
        self._printed = 0

    def poll_status(self, last: str) -> tuple[str, str]:
        """``(status, failure message)`` now; ``(last, "")`` when the call fails."""
        try:
            response = self._api.kernels_status(self._handle)
        except KAGGLE_API_ERRORS as exc:
            logger.debug("Status poll of %s failed; keeping %s: %s", self._handle, last, exc)
            return last, ""
        return status_name(response), failure_message(response)

    def emit_new_lines(self) -> int:
        """Emit the log lines not emitted yet; return how many."""
        try:
            lines = parse_kernel_logs(self._api.kernels_logs(self._handle)).splitlines()
        except KAGGLE_API_ERRORS as exc:
            logger.debug("Log poll of %s failed; retrying next poll: %s", self._handle, exc)
            return 0
        fresh = lines[self._printed :]
        self._printed = max(self._printed, len(lines))
        for line in fresh:
            self._hooks.emit(line)
        return len(fresh)

    def watch(self, initial_status: str = "UNKNOWN") -> WatchResult:
        """Run the loop; the result's ``status`` is terminal, or None when cancelled."""
        result = WatchResult(status=initial_status)
        last = initial_status
        while not self._hooks.cancelled():
            status, failure = self.poll_status(last)
            changed = status != last
            if changed:
                self._hooks.emit(f"[{time.strftime('%H:%M:%S')}] [KAGGLE STATUS] {status}")
            last = status
            result.polls += 1
            result.lines += self.emit_new_lines()
            if status in TERMINAL_STATUSES:
                result.status, result.failure = status, failure
                return result
            if status == RUNNING and self._hooks.on_poll is not None:
                self._hooks.on_poll()
            self._hooks.sleep(self._backoff.next_interval(changed))
        result.status = None
        return result
