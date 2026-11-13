"""Stored member tables of the shards in one directory (the shard index file).

One file holds the tables of every shard in a shard directory::

    {"version": 2, "shards": {"shard-00000.data": {
        "size": 17244160, "mtime_ns": 1762000000000000000,
        "names": [...], "offsets": [...], "sizes": [...]}}}

The same file serves two purposes:

* **Shipped index**: :data:`SHIPPED_INDEX_FILENAME` next to the shards. The suite writes
  it there whenever the shard directory is writable (a local manifold), so a manifold
  uploaded afterwards carries it. The name has no leading dot because Kaggle's dataset
  upload skips hidden files.
* **Scan cache**: the same layout under the suite's writable cache directory when the
  shard directory is read-only (Kaggle's ``/kaggle/input``).

A stored table is used when the shard's size matches and either its modification time
matches too or three header reads agree with the table (:func:`headers_match`). An upload
or a copy changes the modification time, never the size and the member layout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from training.data.containers.tar_offsets import MemberRef, headers_match, shard_stat
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.containers.shard_tables")

SHIPPED_INDEX_FILENAME = "shard_offsets.json"
"""Shard index file written next to the shards (shipped with an uploaded manifold)."""

STORE_VERSION = 2
"""Bumped whenever the file layout changes; files with another version are ignored."""

SHARD_CACHE_DIR_ENV = "LEMGENDARY_SHARD_CACHE_DIR"
"""Overrides the writable folder of scan caches and index locks."""

CACHE_DIRNAME = "shard_offsets"
"""Folder of the scan caches under ``<suite>/.cache``."""


def shard_cache_dir() -> Path:
    """Writable folder of scan caches and index locks.

    ``LEMGENDARY_SHARD_CACHE_DIR`` wins; otherwise ``<suite>/.cache/shard_offsets`` (on
    Kaggle the suite checkout lives in the session-only ``/kaggle/temp``, so the cache is
    not published as kernel output). An unwritable suite folder falls back to the system
    temp dir.
    """
    override = os.environ.get(SHARD_CACHE_DIR_ENV)
    candidates = [Path(override)] if override else []
    candidates += [
        get_project_root() / ".cache" / CACHE_DIRNAME,
        Path(tempfile.gettempdir()) / f"lemgendary_{CACHE_DIRNAME}",
    ]
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("Shard cache folder %s is not usable: %s", folder, exc)
            continue
        if os.access(folder, os.W_OK):
            return folder
    raise OSError(f"No writable shard cache folder among {[str(c) for c in candidates]}")


def cache_key(shard_dir: Path) -> str:
    """File stem identifying ``shard_dir`` inside the cache folder."""
    return hashlib.sha1(str(shard_dir).encode("utf-8")).hexdigest()


def _decode_table(entry: dict[str, Any], shard_size: int) -> list[MemberRef] | None:
    """Member table of a stored entry; ``None`` when malformed or past the shard's end."""
    names, offsets, sizes = entry.get("names"), entry.get("offsets"), entry.get("sizes")
    if not (isinstance(names, list) and isinstance(offsets, list) and isinstance(sizes, list)):
        return None
    if not len(names) == len(offsets) == len(sizes):
        return None
    refs: list[MemberRef] = []
    for name, offset, size in zip(names, offsets, sizes, strict=True):
        if not (isinstance(name, str) and isinstance(offset, int) and isinstance(size, int)):
            return None
        if offset < 0 or size < 0 or offset + size > shard_size:
            return None
        refs.append(MemberRef(name, offset, size))
    return refs


def _encode_table(shard: Path, refs: list[MemberRef]) -> dict[str, Any]:
    """Stored entry of ``shard``'s table."""
    stat = shard_stat(shard)
    return {
        "size": stat.size,
        "mtime_ns": stat.mtime_ns,
        "names": [ref.name for ref in refs],
        "offsets": [ref.offset for ref in refs],
        "sizes": [ref.size for ref in refs],
    }


def _read_store(path: Path) -> dict[str, Any]:
    """Shard entries of a store file ({} when missing, unreadable or another version)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Ignoring unreadable shard index %s: %s", path, exc)
        return {}
    if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
        logger.info("Ignoring shard index %s with another layout version.", path)
        return {}
    shards = data.get("shards")
    return shards if isinstance(shards, dict) else {}


class ShardTableStore:
    """Member tables of one shard directory, read from its shipped index and scan cache."""

    def __init__(self, shard_dir: Path, cache_dir: Path) -> None:
        self.shard_dir = shard_dir
        self.cache_dir = cache_dir
        self._entries: dict[str, Any] = {}
        self._dirty = False
        self.reload()

    @property
    def shipped_path(self) -> Path:
        """The shard index next to the shards."""
        return self.shard_dir / SHIPPED_INDEX_FILENAME

    @property
    def cache_path(self) -> Path:
        """The scan cache of this shard directory in the cache folder."""
        return self.cache_dir / f"{cache_key(self.shard_dir)}.json"

    @property
    def lock_path(self) -> Path:
        """Lock file serialising index builds of this shard directory across processes."""
        return self.cache_dir / f"{cache_key(self.shard_dir)}.lock"

    def reload(self) -> None:
        """Re-read both files (the scan cache's entries win over the shipped index's)."""
        self._entries = {**_read_store(self.shipped_path), **_read_store(self.cache_path)}
        self._dirty = False

    def members(self, shard: Path) -> list[MemberRef] | None:
        """Stored table of ``shard``, or ``None`` when missing, stale or malformed."""
        entry = self._entries.get(shard.name)
        if not isinstance(entry, dict):
            return None
        stat = shard_stat(shard)
        if entry.get("size") != stat.size:
            logger.info("Shard %s changed size since it was indexed; rescanning it.", shard)
            return None
        table = _decode_table(entry, stat.size)
        if table is None:
            logger.warning("Shard index entry for %s is malformed; rescanning it.", shard)
            return None
        if entry.get("mtime_ns") != stat.mtime_ns and not headers_match(shard, table):
            logger.info("Shard %s does not match its index entry; rescanning it.", shard)
            return None
        return table

    def store(self, shard: Path, refs: list[MemberRef]) -> None:
        """Record ``shard``'s freshly scanned table (written by :meth:`save`)."""
        self._entries[shard.name] = _encode_table(shard, refs)
        self._dirty = True

    def save(self) -> Path | None:
        """Write the tables if anything changed; return the file written, if any.

        The shipped index next to the shards is written when that folder is writable,
        otherwise the scan cache.
        """
        if not self._dirty:
            return None
        for name in [n for n in self._entries if not (self.shard_dir / n).is_file()]:
            del self._entries[name]
        payload = json.dumps(
            {"version": STORE_VERSION, "shards": self._entries}, separators=(",", ":")
        )
        for path in (self.shipped_path, self.cache_path):
            if _write_atomic(path, payload):
                self._dirty = False
                return path
        logger.warning(
            "Shard index for %s could not be written; headers are rescanned next time.",
            self.shard_dir,
        )
        return None


def _write_atomic(path: Path, payload: str) -> bool:
    """Write ``payload`` to ``path`` through a temp file; False when the folder refuses."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not write shard index %s: %s", path, exc)
        _discard(tmp)
        return False
    return True


def _discard(path: Path) -> None:
    """Remove a leftover temp file, logging when that fails."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Could not remove temp file %s: %s", path, exc)
