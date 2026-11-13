"""Member tables of a shard list: stored index first, one parallel scan for the rest.

For each shard directory the tables come from :class:`ShardTableStore` (the shipped
index next to the shards, then the scan cache). Shards without a valid stored table are
scanned once, under a file lock per shard directory:

* the first process (one DDP rank, or the train and validation readers of one run) takes
  the lock, scans the missing shards and saves the tables;
* every other process blocks on the lock meanwhile, then re-reads the saved tables and
  scans nothing.

The lock is an operating-system file lock (``filelock``), released when its holder exits,
so a crashed builder never leaves the others waiting. It needs no process group, so it
works before ``torch.distributed`` is set up and has no collective timeout.

The first missing shard is scanned alone as a probe. When its headers came slower than
:data:`LATENCY_BOUND_SECONDS` each, the storage is latency-bound (Kaggle's network-backed
``/kaggle/input``: about 0.7 ms per header) and the rest is scanned on :data:`SCAN_THREADS`
threads, which overlap the waits. On a local disk the scan is CPU-bound, where threads
only contend for the interpreter lock, so it stays on one thread.
"""

from __future__ import annotations

import logging
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock, Timeout

from training.data.containers import tar_offsets
from training.data.containers.shard_tables import ShardTableStore, shard_cache_dir
from training.data.containers.tar_offsets import MemberRef

logger = logging.getLogger("lemtrain.containers.member_tables")

SCAN_THREADS = 16
"""Shards scanned at once on latency-bound storage."""

LATENCY_BOUND_SECONDS = 100e-6
"""Seconds per member header above which the probe calls the storage latency-bound (a
local disk scans a header in about 30 microseconds)."""

MemberTables = dict[Path, list[MemberRef] | None]
"""Table per shard; ``None`` marks a shard read through ``tarfile`` (compressed)."""


def _stored_tables(store: ShardTableStore, shards: list[Path]) -> MemberTables:
    """Tables of ``shards`` that the store holds and that still match the shard."""
    tables: MemberTables = {}
    for shard in shards:
        try:
            table = store.members(shard)
        except OSError as exc:
            logger.warning("Failed reading shard %s: %s", shard, exc)
            continue
        if table is not None:
            tables[shard] = table
    return tables


ScanResult = tuple[Path, list[MemberRef] | None, BaseException | None]


def _scan_one(shard: Path) -> ScanResult:
    """Scan one shard; a damaged shard comes back with its error instead of a table."""
    try:
        return shard, tar_offsets.scan_tar_members(shard), None
    except (OSError, tarfile.TarError, EOFError) as exc:
        return shard, None, exc


def scan_workers(probe: ScanResult, seconds: float, remaining: int) -> int:
    """Threads for the ``remaining`` shards after a probe scan that took ``seconds``."""
    table = probe[1]
    per_header = seconds / max(1, len(table) if table else 1)
    if per_header <= LATENCY_BOUND_SECONDS:
        return 1
    return max(1, min(SCAN_THREADS, remaining))


def _record(store: ShardTableStore, tables: MemberTables, result: ScanResult) -> None:
    """Keep one scan result in ``tables`` and ``store`` (a damaged shard is logged)."""
    shard, table, error = result
    if error is not None:
        logger.warning("Failed opening tar file %s: %s", shard, error)
        return
    tables[shard] = table
    if table is not None:
        store.store(shard, table)


def _scan_missing(store: ShardTableStore, missing: list[Path]) -> MemberTables:
    """Scan ``missing`` (a probe, then the rest), record the tables and save ``store``."""
    print(
        f"[CONTAINER] Indexing {len(missing)} shard(s) in {store.shard_dir} "
        f"(no stored shard index for them)...",
        flush=True,
    )
    started = time.perf_counter()
    tables: MemberTables = {}
    probe = _scan_one(missing[0])
    _record(store, tables, probe)
    rest = missing[1:]
    workers = scan_workers(probe, time.perf_counter() - started, len(rest))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_scan_one, rest):
                _record(store, tables, result)
    else:
        for shard in rest:
            _record(store, tables, _scan_one(shard))
    written = store.save()
    print(
        f"[CONTAINER] Indexed {len(tables)} shard(s) in {time.perf_counter() - started:.1f} s"
        f" on {workers} thread(s){f'; saved to {written}' if written else ''}.",
        flush=True,
    )
    return tables


def _acquire(lock: FileLock, shard_dir: Path) -> None:
    """Take ``lock``; say so when another process holds it (it is building the index)."""
    try:
        lock.acquire(timeout=0)
    except Timeout:
        print(
            f"[CONTAINER] Waiting for another process to index the shards in {shard_dir}...",
            flush=True,
        )
        lock.acquire()


def _dir_tables(shard_dir: Path, shards: list[Path]) -> MemberTables:
    """Tables of the shards of one directory (stored, else scanned under the lock)."""
    store = ShardTableStore(shard_dir, shard_cache_dir())
    tables = _stored_tables(store, shards)
    if len(tables) == len(shards):
        return tables
    lock = FileLock(str(store.lock_path))
    _acquire(lock, shard_dir)
    try:
        store.reload()
        tables = _stored_tables(store, shards)
        missing = [shard for shard in shards if shard not in tables]
        if missing:
            tables.update(_scan_missing(store, missing))
    finally:
        lock.release()
    return tables


def load_member_tables(shards: list[Path]) -> MemberTables:
    """Member tables of ``shards``; shards that fail to open are left out and logged."""
    by_dir: dict[Path, list[Path]] = {}
    for shard in shards:
        by_dir.setdefault(shard.parent, []).append(shard)
    tables: MemberTables = {}
    for shard_dir, dir_shards in by_dir.items():
        tables.update(_dir_tables(shard_dir, dir_shards))
    return tables
