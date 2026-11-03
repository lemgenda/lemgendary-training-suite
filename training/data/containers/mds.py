"""MDS (MosaicML Streaming Dataset) container reader with Zstd decompression support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from training.data.containers.base import Sample
from training.utils.lazy_imports import load_streaming

logger = logging.getLogger("lemtrain.containers.mds")

# Failure modes of the optional `streaming` library when opening a local shard directory.
_STREAMING_INIT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
    TypeError,
)


def _as_bytes(value: Any) -> bytes | None:
    return bytes(value) if isinstance(value, (bytes, bytearray)) else None


def _record_sample(record: dict[str, Any], index: int) -> Sample:
    """``Sample`` from one streaming MDS record (field aliases resolved)."""
    meta = {"index": index}
    if isinstance(record.get("metadata"), dict):
        meta.update(record["metadata"])
    return Sample(
        name=str(record.get("name") or record.get("id") or f"mds_sample_{index:08d}"),
        image_bytes=_as_bytes(record.get("image") or record.get("image_bytes") or b"") or b"",
        image_format=str(record.get("image_format", "webp")),
        target_bytes=_as_bytes(record.get("target") or record.get("target_bytes")),
        mask_bytes=_as_bytes(record.get("mask") or record.get("mask_bytes")),
        label=record.get("label"),
        metadata=meta,
    )


class MdsReader:
    """Reader for MosaicML Streaming (MDS) sharded datasets."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate MDS directory across candidate locations
        cand_dirs = [
            self.root / "mds" / self.split,
            self.root / self.split / "mds",
            self.root / "mds",
            self.root / self.split,
            self.root,
        ]
        self.mds_dir = self.root
        for d in cand_dirs:
            if d.exists() and d.is_dir() and (d / "index.json").exists():
                self.mds_dir = d
                break

        index_file = self.mds_dir / "index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"MDS index.json not found in '{self.mds_dir}'.")

        try:
            index_data = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed parsing MDS index.json at '{index_file}': {exc}") from exc

        shards: list[dict[str, Any]] = index_data.get("shards", [])
        self._sample_count = 0
        self._shard_offsets: list[
            tuple[int, int, dict[str, Any]]
        ] = []  # (start_idx, end_idx, shard)

        for shard in shards:
            samples = shard.get("samples", 0)
            self._shard_offsets.append((self._sample_count, self._sample_count + samples, shard))
            self._sample_count += samples

        self._streaming_ds: Any = None
        self._init_streaming()

    def _init_streaming(self) -> None:
        """Attempt initializing via official streaming library if available."""
        try:
            # Optional MosaicML dependency; absent installs fall back to the raw index.
            streaming_pkg = load_streaming()

            local_ds_cls = getattr(streaming_pkg, "LocalDataset", None)
            streaming_cls = getattr(streaming_pkg, "StreamingDataset", None)
            if local_ds_cls is not None:
                try:
                    self._streaming_ds = local_ds_cls(local=str(self.mds_dir))
                    return
                except _STREAMING_INIT_ERRORS as exc:
                    logger.debug(
                        "LocalDataset initialization failed, trying StreamingDataset: %s", exc
                    )
            if streaming_cls is not None:
                self._streaming_ds = streaming_cls(
                    local=str(self.mds_dir),
                    split=None,
                    shuffle=False,
                )
        except ImportError as exc:
            logger.debug("Streaming library not available, operating via raw index: %s", exc)
            self._streaming_ds = None
        except _STREAMING_INIT_ERRORS as exc:
            logger.warning(
                "Streaming library initialization failed for %s, operating via raw index: %s",
                self.mds_dir,
                exc,
            )
            self._streaming_ds = None

    def __len__(self) -> int:
        if self._streaming_ds is not None:
            return len(self._streaming_ds)
        return self._sample_count

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(f"MDS sample index {index} out of range (0..{len(self) - 1}).")

        if self._streaming_ds is not None:
            return _record_sample(self._streaming_ds[index], index)
        return self._index_sample(index)

    def _index_sample(self, index: int) -> Sample:
        """Placeholder sample from the index metadata (no streaming dataset available)."""
        target_shard: dict[str, Any] | None = None
        local_idx = 0
        for start, end, shard in self._shard_offsets:
            if start <= index < end:
                target_shard = shard
                local_idx = index - start
                break

        shard_name = (
            target_shard.get("raw_data", {}).get("basename", "shard") if target_shard else "shard"
        )
        return Sample(
            name=f"mds_sample_{index:08d}",
            image_bytes=b"",
            image_format="webp",
            target_bytes=None,
            label=None,
            metadata={"shard": shard_name, "local_index": local_idx, "index": index},
        )

    def close(self) -> None:
        """Release streaming dataset references."""
        self._streaming_ds = None
