"""Bounded, per-process LRU pool of open shard handles.

A reader that keeps every shard it touched open holds ``num_workers * num_shards``
descriptors (270 shards on the largest manifolds). The pool keeps at most ``max_open``
handles per process and closes the least recently used one when a new shard is opened.

Handles never cross a process boundary: pickling keeps only the pool's settings, and a
pool used in a forked child closes the inherited handles first (they share the parent's
file offset, so seeking in the child would move the parent's reads). A pool that is
garbage-collected without ``close()`` (datasets never close their readers) closes its
handles through a ``weakref.finalize`` hook.
"""

from __future__ import annotations

import logging
import os
import tarfile
import weakref
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Generic, TypeVar

logger = logging.getLogger("lemtrain.containers.handle_pool")

DEFAULT_MAX_OPEN_HANDLES = 16
"""Open shard handles kept per process (per dataloader worker)."""


HandleT = TypeVar("HandleT", IO[bytes], tarfile.TarFile)
"""Pooled handle kinds: binary files (offset reads) and ``tarfile`` archives."""


class HandlePool(Generic[HandleT]):
    """LRU map of shard path to open handle, capped at ``max_open`` per process."""

    def __init__(
        self, opener: Callable[[Path], HandleT], max_open: int = DEFAULT_MAX_OPEN_HANDLES
    ) -> None:
        if max_open < 1:
            raise ValueError(f"max_open must be at least 1, got {max_open}")
        self._opener = opener
        self.max_open = int(max_open)
        self._handles: OrderedDict[Path, HandleT] = OrderedDict()
        self._pid = os.getpid()
        weakref.finalize(self, _close_all, self._handles)

    def __len__(self) -> int:
        return len(self._handles)

    def get(self, path: Path) -> HandleT:
        """Open handle of ``path``, opening it (and evicting the LRU handle) when needed."""
        self._claim_process()
        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle
        handle = self._opener(path)
        self._handles[path] = handle
        while len(self._handles) > self.max_open:
            evicted_path, evicted = self._handles.popitem(last=False)
            _close(evicted_path, evicted)
        return handle

    def discard(self, path: Path) -> None:
        """Close and forget the handle of ``path`` (after a failed read)."""
        handle = self._handles.pop(path, None)
        if handle is not None:
            _close(path, handle)

    def close(self) -> None:
        """Close every handle this process opened."""
        _close_all(self._handles)

    def _claim_process(self) -> None:
        current = os.getpid()
        if current != self._pid:
            self.close()
            self._pid = current

    def __getstate__(self) -> dict[str, Any]:
        return {"opener": self._opener, "max_open": self.max_open}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._opener = state["opener"]
        self.max_open = int(state["max_open"])
        self._handles = OrderedDict()
        self._pid = os.getpid()
        weakref.finalize(self, _close_all, self._handles)


def _close(path: Path, handle: IO[bytes] | tarfile.TarFile) -> None:
    try:
        handle.close()
    except OSError as exc:
        logger.debug("Error closing shard handle %s: %s", path, exc)


def _close_all(handles: OrderedDict[Path, Any]) -> None:
    """Close and forget every handle in ``handles`` (oldest first)."""
    while handles:
        path, handle = handles.popitem(last=False)
        _close(path, handle)
