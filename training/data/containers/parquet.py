"""Parquet container reader wrapping PyArrow with row-group level caching."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import pyarrow.parquet as pq

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.parquet")


@dataclass
class _CachedRowGroup:
    """Cached table slice for a single row group."""

    file_index: int
    group_index: int
    start_row: int
    end_row: int
    data: dict[str, list[Any]]


class ParquetReader:
    """High-performance Parquet container reader with LRU row-group caching."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Discover parquet files
        if self.root.is_file() and self.root.suffix == ".parquet":
            self.files = [self.root]
        elif (self.root / self.split).exists() and (self.root / self.split).is_dir():
            self.files = sorted((self.root / self.split).glob("*.parquet"))
        else:
            self.files = sorted(self.root.glob("*.parquet"))

        if not self.files:
            raise FileNotFoundError(f"No Parquet files discovered in '{self.root}' for split '{self.split}'.")

        # Map global indices to (file_idx, row_group_idx, local_row_idx)
        self.file_metadata: list[pq.FileMetaData] = []
        self.index_map: list[tuple[int, int, int]] = []  # (file_idx, row_group_idx, row_in_group)
        self._total_rows = 0

        for f_idx, f_path in enumerate(self.files):
            meta = pq.read_metadata(f_path)
            self.file_metadata.append(meta)
            for rg_idx in range(meta.num_row_groups):
                rg_rows = meta.row_group(rg_idx).num_rows
                for r in range(rg_rows):
                    self.index_map.append((f_idx, rg_idx, r))
                self._total_rows += rg_rows

        self._cache: _CachedRowGroup | None = None

    def __len__(self) -> int:
        return self._total_rows

    def _get_row(self, file_idx: int, rg_idx: int, row_in_group: int) -> dict[str, Any]:
        """Fetch row using row-group cache."""
        if self._cache is None or self._cache.file_index != file_idx or self._cache.group_index != rg_idx:
            parquet_file = pq.ParquetFile(self.files[file_idx])
            table = parquet_file.read_row_group(rg_idx)
            data_dict = {col: table[col].to_pylist() for col in table.column_names}
            self._cache = _CachedRowGroup(
                file_index=file_idx,
                group_index=rg_idx,
                start_row=0,
                end_row=len(next(iter(data_dict.values()))),
                data=data_dict,
            )

        return {col: self._cache.data[col][row_in_group] for col in self._cache.data}

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= self._total_rows:
            raise IndexError(f"Parquet row index {index} out of range (0..{self._total_rows - 1}).")

        file_idx, rg_idx, row_in_rg = self.index_map[index]
        row = self._get_row(file_idx, rg_idx, row_in_rg)

        name = str(row.get("id") or row.get("name") or f"parquet_sample_{index:08d}")

        # Check if raw image bytes are stored in column
        image_bytes: bytes = b""
        img_fmt = "raw"
        for col_name in ["image", "image_bytes", "data"]:
            if col_name in row and isinstance(row[col_name], (bytes, bytearray)):
                image_bytes = bytes(row[col_name])
                img_fmt = str(row.get("image_format", "webp"))
                break

        # Targets / Labels
        target_bytes: bytes | None = None
        for tgt_name in ["target", "target_bytes", "ground_truth"]:
            if tgt_name in row and isinstance(row[tgt_name], (bytes, bytearray)):
                target_bytes = bytes(row[tgt_name])
                break

        label = row.get("label") or row.get("target") or row.get("signal")

        return Sample(
            name=name,
            image_bytes=image_bytes,
            image_format=img_fmt,
            target_bytes=target_bytes,
            label=label,
            metadata=row,
        )

    def close(self) -> None:
        """Clear cached row group."""
        self._cache = None
