"""MDS (MosaicML Streaming Dataset) container reader with Zstd decompression support."""

import json
import logging
from pathlib import Path
from typing import Any

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.mds")


class MdsReader:
    """Reader for MosaicML Streaming (MDS) sharded datasets."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate MDS directory
        split_dir = self.root / self.split
        self.mds_dir = split_dir if split_dir.exists() and split_dir.is_dir() else self.root

        self.index_file = self.mds_dir / "index.json"
        if not self.index_file.exists():
            raise FileNotFoundError(f"MDS index.json not found in '{self.mds_dir}'.")

        try:
            self.index_data = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed parsing MDS index.json at '{self.index_file}': {exc}") from exc

        self.shards: list[dict[str, Any]] = self.index_data.get("shards", [])
        self._sample_count = 0
        self._shard_offsets: list[tuple[int, int, dict[str, Any]]] = []  # (start_idx, end_idx, shard)

        for shard in self.shards:
            samples = shard.get("samples", 0)
            self._shard_offsets.append((self._sample_count, self._sample_count + samples, shard))
            self._sample_count += samples

        self._streaming_ds: Any = None
        self._init_streaming()

    def _init_streaming(self) -> None:
        """Attempt initializing via official streaming library if available."""
        try:
            import importlib
            streaming_pkg = importlib.import_module("streaming")
            streaming_cls = getattr(streaming_pkg, "StreamingDataset", None)
            if streaming_cls is not None:
                self._streaming_ds = streaming_cls(
                    local=str(self.mds_dir),
                    split=None,
                    shuffle=False,
                )
        except (ImportError, ModuleNotFoundError, AttributeError):
            logger.debug("Streaming library not installed. Operating via MDS raw index parser.")
            self._streaming_ds = None

    def __len__(self) -> int:
        if self._streaming_ds is not None:
            return len(self._streaming_ds)
        return self._sample_count

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(f"MDS sample index {index} out of range (0..{len(self) - 1}).")

        if self._streaming_ds is not None:
            record = self._streaming_ds[index]
            name = str(record.get("name") or record.get("id") or f"mds_sample_{index:08d}")
            image_bytes = record.get("image") or record.get("image_bytes") or b""
            fmt = str(record.get("image_format", "webp"))
            target_bytes = record.get("target") or record.get("target_bytes")
            mask_bytes = record.get("mask") or record.get("mask_bytes")
            label = record.get("label")
            return Sample(
                name=name,
                image_bytes=bytes(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else b"",
                image_format=fmt,
                target_bytes=bytes(target_bytes) if isinstance(target_bytes, (bytes, bytearray)) else None,
                mask_bytes=bytes(mask_bytes) if isinstance(mask_bytes, (bytes, bytearray)) else None,
                label=label,
                metadata={"index": index},
            )

        # Fallback index metadata record
        target_shard: dict[str, Any] | None = None
        local_idx = 0
        for start, end, shard in self._shard_offsets:
            if start <= index < end:
                target_shard = shard
                local_idx = index - start
                break

        shard_name = target_shard.get("raw_data", {}).get("basename", "shard") if target_shard else "shard"
        return Sample(
            name=f"mds_sample_{index:08d}",
            image_bytes=b"",
            image_format="webp",
            target_bytes=None,
            label=None,
            metadata={"shard": shard_name, "local_index": local_idx},
        )

    def close(self) -> None:
        """Release streaming dataset references."""
        self._streaming_ds = None
