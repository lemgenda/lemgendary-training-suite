"""Where a container sample came from, for warnings the dataset side can act on.

A decode warning that names only ``dataset[index]`` cannot be traced back to the packed
data: the index depends on the shard order and the reader. :func:`describe_sample_origin`
names the sample key, its member, its shard and what the payload looks like, so the
dataset compiler can find and repair the record.
"""

from __future__ import annotations

from typing import Any

from training.data.containers.base import Sample

PAYLOAD_HEAD_BYTES = 8
"""Leading payload bytes shown (as hex) in the description: enough for any image magic."""


def _shard_of(reader: Any, index: int) -> str | None:
    """The reader's shard of ``index`` (``shard_of``), or ``None`` when it has no shards."""
    shard_of = getattr(reader, "shard_of", None)
    shard = shard_of(index) if callable(shard_of) else None
    return None if shard is None else str(shard)


def describe_sample_origin(reader: Any, index: int, sample: Sample, payload: bytes) -> str:
    """``key ..., member ..., N bytes, head ..., shard ...`` of one sample's ``payload``.

    The member comes from the sample's metadata (``member``, ``file`` or ``path``, whichever
    the reader records); the shard from the reader's ``shard_of`` when it has one.
    """
    parts = [f"key '{sample.name}'"]
    member = next(
        (sample.metadata[k] for k in ("member", "file", "path") if k in sample.metadata), None
    )
    if member is not None:
        parts.append(f"member '{member}'")
    parts.append(f"{len(payload)} bytes, head {payload[:PAYLOAD_HEAD_BYTES].hex() or '-'}")
    shard = _shard_of(reader, index)
    if shard is not None:
        parts.append(f"shard '{shard}'")
    return ", ".join(parts)
