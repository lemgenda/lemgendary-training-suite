"""LitData container reader optimized for variable-shape array streaming."""

import importlib
import logging
from pathlib import Path
from typing import Any

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.litdata")


class LitDataReader:
    """Reader for PyTorch-Lightning LitData chunked streaming datasets."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate LitData split directory across candidate locations
        cand_dirs = [
            self.root / "litdata" / self.split,
            self.root / self.split / "litdata",
            self.root / "litdata",
            self.root / self.split,
            self.root,
        ]
        self.data_dir = self.root
        for d in cand_dirs:
            if d.exists() and d.is_dir():
                try:
                    if any(d.glob("*.bin")) or (d / "index.json").exists():
                        self.data_dir = d
                        break
                except OSError:
                    pass

        self._lit_ds: Any = None
        self._sample_count = 0
        self._chunks: list[Path] = []
        self._init_litdata()

    def _init_litdata(self) -> None:
        """Initialize LitData StreamingDataset if library is present and valid index exists."""
        try:
            litdata_pkg = importlib.import_module("litdata")
            streaming_cls = getattr(litdata_pkg, "StreamingDataset", None)
            if streaming_cls is not None and (self.data_dir / "index.json").exists():
                self._lit_ds = streaming_cls(input_dir=str(self.data_dir))
                self._sample_count = len(self._lit_ds)
                return
        except Exception as exc:
            logger.debug("LitData library initialization failed: %s", exc)

        # Fallback: scan for chunk / bin files in data_dir
        self._chunks = sorted(set(self.data_dir.glob("*.bin")))
        self._sample_count = len(self._chunks)

    def __len__(self) -> int:
        if self._lit_ds is not None:
            return len(self._lit_ds)
        return self._sample_count

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(f"LitData index {index} out of range (0..{len(self) - 1}).")

        if self._lit_ds is not None:
            item = self._lit_ds[index]
            if isinstance(item, dict):
                name = str(item.get("id") or item.get("name") or f"lit_sample_{index:08d}")
                image_bytes = item.get("image") or item.get("image_bytes") or b""
                target_bytes = item.get("target") or item.get("target_bytes")
                mask_bytes = item.get("mask") or item.get("mask_bytes")
                label = item.get("label")
                meta = {"index": index}
                if isinstance(item.get("metadata"), dict):
                    meta.update(item["metadata"])
                return Sample(
                    name=name,
                    image_bytes=bytes(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else b"",
                    image_format=str(item.get("image_format", "webp")),
                    target_bytes=bytes(target_bytes) if isinstance(target_bytes, (bytes, bytearray)) else None,
                    mask_bytes=bytes(mask_bytes) if isinstance(mask_bytes, (bytes, bytearray)) else None,
                    label=label,
                    metadata=meta,
                )

        return Sample(
            name=f"lit_sample_{index:08d}",
            image_bytes=b"",
            image_format="webp",
            metadata={"index": index},
        )

    def close(self) -> None:
        """Release underlying references."""
        self._lit_ds = None
