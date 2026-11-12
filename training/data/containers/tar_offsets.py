"""Byte-offset tables of uncompressed tar shards and their persistent cache.

``tarfile.extractfile(name)`` resolves ``name`` with a reverse linear scan over every
member of the archive, and the first lookup on a fresh handle reads every tar header of
the shard. A WebDataset shard holds 5,000 to 10,000 members and each sample needs two to
five lookups, so readers paid that scan per member read and a full header scan per
worker, per shard, after every loader rebuild.

An uncompressed tar stores each member's payload contiguously at ``TarInfo.offset_data``.
:func:`scan_tar_members` records ``(name, offset, size)`` for every regular member once;
readers then fetch a payload with one ``seek`` and one ``read``. :class:`ShardOffsetCache`
persists the tables next to the shards (or in a temp-dir fallback when the shard folder is
read-only, as on Kaggle's ``/kaggle/input``), keyed by shard file name and invalidated by
the shard's size and modification time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger("lemtrain.containers.tar_offsets")

CACHE_FILENAME = ".shard_offsets.json"
"""Cache file written into the shard directory."""

CACHE_VERSION = 1
"""Bumped whenever the cache layout changes; other versions are ignored."""

FALLBACK_CACHE_DIRNAME = "lemgendary_shard_offsets"
"""Folder under the system temp dir used when the shard directory is not writable."""


class MemberRef(NamedTuple):
    """Location of one regular tar member's payload."""

    name: str
    offset: int
    size: int


def shard_fingerprint(path: Path) -> list[int]:
    """``[size, mtime_ns]`` of ``path``; a change in either invalidates its cached table."""
    stat = path.stat()
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def scan_tar_members(path: Path) -> list[MemberRef] | None:
    """Offsets of every regular member of an uncompressed tar, in archive order.

    Returns ``None`` when the shard cannot be read by offset (compressed or holding
    sparse members); such shards are read through ``tarfile`` instead. A damaged
    archive raises ``tarfile.TarError`` / ``OSError`` like ``tarfile.open`` does.
    """
    try:
        with tarfile.open(path, mode="r:") as tf:
            members = tf.getmembers()
    except tarfile.ReadError as exc:
        logger.debug("Shard %s is not an uncompressed tar (%s); using tarfile reads.", path, exc)
        return None
    refs: list[MemberRef] = []
    for member in members:
        if not member.isfile():
            continue
        if member.issparse():
            logger.debug("Shard %s holds sparse member %s; using tarfile reads.", path, member.name)
            return None
        refs.append(MemberRef(member.name, int(member.offset_data), int(member.size)))
    return refs


def _fallback_cache_path(shard_dir: Path) -> Path:
    """Temp-dir cache path unique to ``shard_dir``."""
    digest = hashlib.sha1(str(shard_dir).encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / FALLBACK_CACHE_DIRNAME / f"{digest}.json"


def _valid_table(raw: Any, shard_size: int) -> list[MemberRef] | None:
    """Decode a cached member table; ``None`` when malformed or past the shard's end."""
    if not isinstance(raw, list):
        return None
    refs: list[MemberRef] = []
    for item in raw:
        if not (isinstance(item, list) and len(item) == 3):
            return None
        name, offset, size = item
        if not (isinstance(name, str) and isinstance(offset, int) and isinstance(size, int)):
            return None
        if offset < 0 or size < 0 or offset + size > shard_size:
            return None
        refs.append(MemberRef(name, offset, size))
    return refs


class ShardOffsetCache:
    """Persistent member tables of the shards in one directory."""

    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = shard_dir
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def candidate_paths(self) -> tuple[Path, Path]:
        """Primary (next to the shards) and fallback (temp dir) cache locations."""
        return self.shard_dir / CACHE_FILENAME, _fallback_cache_path(self.shard_dir)

    def _load(self) -> None:
        for path in self.candidate_paths():
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                logger.warning("Ignoring unreadable shard offset cache %s: %s", path, exc)
                continue
            if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
                logger.info("Ignoring shard offset cache %s with another layout version.", path)
                continue
            shards = data.get("shards")
            if isinstance(shards, dict):
                self._entries = shards
                return

    def members(self, shard: Path) -> list[MemberRef] | None:
        """Cached table of ``shard``, or ``None`` when missing or stale."""
        entry = self._entries.get(shard.name)
        if not isinstance(entry, dict):
            return None
        fingerprint = shard_fingerprint(shard)
        if entry.get("fingerprint") != fingerprint:
            logger.info("Shard %s changed since it was indexed; rescanning its headers.", shard)
            return None
        table = _valid_table(entry.get("members"), fingerprint[0])
        if table is None:
            logger.warning("Shard offset cache entry for %s is malformed; rescanning.", shard)
        return table

    def store(self, shard: Path, members: list[MemberRef]) -> None:
        """Record ``shard``'s freshly scanned table (written by :meth:`save`)."""
        self._entries[shard.name] = {
            "fingerprint": shard_fingerprint(shard),
            "members": [list(ref) for ref in members],
        }
        self._dirty = True

    def save(self) -> Path | None:
        """Write the tables if anything changed; return the file written, if any."""
        if not self._dirty:
            return None
        for name in [n for n in self._entries if not (self.shard_dir / n).is_file()]:
            del self._entries[name]
        payload = json.dumps(
            {"version": CACHE_VERSION, "shards": self._entries}, separators=(",", ":")
        )
        for path in self.candidate_paths():
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, path)
            except OSError as exc:
                logger.debug("Could not write shard offset cache %s: %s", path, exc)
                _discard(tmp)
                continue
            self._dirty = False
            return path
        logger.warning(
            "Shard offset cache for %s could not be written; headers are rescanned next run.",
            self.shard_dir,
        )
        return None


def _discard(path: Path) -> None:
    """Remove a leftover temp file, logging when that fails."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Could not remove temp file %s: %s", path, exc)


def load_member_tables(shards: list[Path]) -> dict[Path, list[MemberRef] | None]:
    """Member tables of ``shards`` from the cache, scanning (and caching) the rest.

    A ``None`` table marks a shard that must be read through ``tarfile``. Shards that
    fail to open are left out and logged.
    """
    caches: dict[Path, ShardOffsetCache] = {}
    tables: dict[Path, list[MemberRef] | None] = {}
    for shard in shards:
        cache = caches.get(shard.parent)
        if cache is None:
            cache = caches[shard.parent] = ShardOffsetCache(shard.parent)
        try:
            table = cache.members(shard)
            if table is None:
                table = scan_tar_members(shard)
                if table is not None:
                    cache.store(shard, table)
        except (OSError, tarfile.TarError, EOFError) as exc:
            logger.warning("Failed opening tar file %s: %s", shard, exc)
            continue
        tables[shard] = table
    for cache in caches.values():
        cache.save()
    return tables
