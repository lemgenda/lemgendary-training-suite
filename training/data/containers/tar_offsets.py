"""Byte-offset tables of uncompressed tar shards: scanning and header spot checks.

``tarfile.extractfile(name)`` resolves ``name`` with a reverse linear scan over every
member of the archive, and the first lookup on a fresh handle reads every tar header of
the shard. A WebDataset shard holds 5,000 to 10,000 members and each sample needs two to
five lookups, so readers paid that scan per member read and a full header scan per
worker, per shard, after every loader rebuild.

An uncompressed tar stores each member's payload contiguously at ``TarInfo.offset_data``.
:func:`scan_tar_members` records ``(name, offset, size)`` for every regular member once;
readers then fetch a payload with one ``seek`` and one ``read``. The tables are stored by
:mod:`training.data.containers.shard_tables` and loaded by
:mod:`training.data.containers.member_tables`.

A full scan reads every member header, one small random read per member: about 4 million
reads for a 1.3 million sample manifold, which took 48 minutes on Kaggle's network-backed
``/kaggle/input``. :func:`headers_match` checks a stored table against the shard with
three header reads instead.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("lemtrain.containers.tar_offsets")

HEADER_SIZE = tarfile.BLOCKSIZE
"""Size of one tar header block; a member's header ends where its payload starts."""


class MemberRef(NamedTuple):
    """Location of one regular tar member's payload."""

    name: str
    offset: int
    size: int


class ShardStat(NamedTuple):
    """Size and modification time of a shard file (a stored table's validity key)."""

    size: int
    mtime_ns: int


def shard_stat(path: Path) -> ShardStat:
    """``ShardStat`` of ``path``; a change in either field means the shard changed."""
    stat = path.stat()
    return ShardStat(int(stat.st_size), int(stat.st_mtime_ns))


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


def _spot_check_refs(refs: list[MemberRef]) -> list[MemberRef]:
    """First, middle and last member of a table (fewer for short tables)."""
    if not refs:
        return []
    picks = {0, len(refs) // 2, len(refs) - 1}
    return [refs[i] for i in sorted(picks)]


def _header_matches(header: bytes, ref: MemberRef) -> bool:
    """Whether the tar header block right before ``ref``'s payload describes ``ref``.

    A long name lives in a preceding PAX or GNU record, so the block's own name field may
    hold only its start: the name must be a non-empty prefix of ``ref.name``.
    """
    try:
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
    except tarfile.HeaderError as exc:
        logger.debug("No valid tar header before member %s: %s", ref.name, exc)
        return False
    return info.size == ref.size and bool(info.name) and ref.name.startswith(info.name)


def headers_match(path: Path, refs: list[MemberRef]) -> bool:
    """Whether the shard's headers agree with ``refs`` at the first, middle and last member.

    Three 512-byte reads; any mismatch, short read or I/O error means the table does not
    describe this shard.
    """
    try:
        with open(path, "rb") as handle:
            for ref in _spot_check_refs(refs):
                if ref.offset < HEADER_SIZE:
                    return False
                handle.seek(ref.offset - HEADER_SIZE)
                header = handle.read(HEADER_SIZE)
                if len(header) != HEADER_SIZE or not _header_matches(header, ref):
                    return False
    except OSError as exc:
        logger.warning("Could not check the tar headers of %s: %s", path, exc)
        return False
    return True
