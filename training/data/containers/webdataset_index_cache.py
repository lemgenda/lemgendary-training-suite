"""Persistent per-shard sample index cache for WebDataset tar shards.

Indexing a tar shard reads every member header, which dominates start-up on large
datasets. The result is stored next to the shard and reused while the shard's size and
modification time are unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("lemtrain.containers.webdataset")

CACHE_SUFFIX = ".index.json"
CACHE_VERSION = 1

# One sample: (image, target, mask, text, json) tar member names; absent roles are None.
MemberRefs = tuple[str | None, str | None, str | None, str | None, str | None]


def _cache_path(shard: Path) -> Path:
    return shard.with_name(shard.name + CACHE_SUFFIX)


def _fingerprint(shard: Path) -> dict[str, int]:
    stat = shard.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_cached_index(shard: Path) -> list[MemberRefs] | None:
    """The cached sample index of ``shard``, or None when missing, stale or unreadable."""
    path = _cache_path(shard)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["version"] != CACHE_VERSION or payload["shard"] != _fingerprint(shard):
            return None
        return [tuple(refs) for refs in payload["samples"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable shard index cache %s: %s", path, exc)
        return None


def store_cached_index(shard: Path, samples: list[MemberRefs]) -> None:
    """Write the sample index of ``shard``; a read-only location only costs the cache."""
    path = _cache_path(shard)
    payload = {"version": CACHE_VERSION, "shard": _fingerprint(shard), "samples": samples}
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("Shard index cache %s could not be written: %s", path, exc)

