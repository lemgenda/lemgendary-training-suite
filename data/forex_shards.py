"""Forex shard loading: the Parquet row-group LRU cache and chunked .npy shards."""

from __future__ import annotations

import bisect
import logging
import os
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch

from data.forex_spec import (
    shard_file_paths,
)

logger = logging.getLogger("lemtrain.data.forex_dataset")


def safe_from_numpy(arr: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a writable, C-contiguous float32 PyTorch tensor."""
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"] and arr.flags["WRITEABLE"]:
        return torch.from_numpy(arr)
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))


class ParquetRowGroupCache:
    """
    Process-safe LRU cache for Parquet row groups.

    Row groups are decoded to per-row numpy arrays at cache-fill time, not
    on every access. Under the RowGroupAwareSampler this turns a cache hit
    from a ~5 ms pyarrow scalar extraction into a ~3 microsecond numpy
    slice.

    Memory: each cached row group holds ~5000 decoded rows. Row sizes vary
    by timeframe (feature width is fixed at 14, seq_len varies 90..512).
    Worst case ~5000 * 512 * 14 * 4 bytes ≈ 140 MB per group. With 12
    groups that's ~1.7 GB peak. On Kaggle (30 GB RAM) this is negligible;
    on a 4 GB local machine, consider lowering max_cached_groups to 4.
    """

    def __init__(self, parquet_path: str, max_cached_groups: int = 12):
        self.parquet_path = parquet_path
        self.max_cached = max_cached_groups
        self.pf = pq.ParquetFile(parquet_path)
        self._cache = {}
        self._lru_order = []

        metadata = self.pf.metadata
        self.rg_starts = []
        curr = 0
        for i in range(self.pf.num_row_groups):
            self.rg_starts.append(curr)
            curr += metadata.row_group(i).num_rows
        self.total_rows = curr

    @property
    def num_row_groups(self) -> int:
        """Number of row groups in the Parquet file."""
        return len(self.rg_starts)

    @property
    def metadata(self):
        """Parquet file metadata."""
        return self.pf.metadata

    def row_group_of_rows(self) -> np.ndarray:
        """Return the row-group number of every row (``RowGroupAwareSampler`` input)."""
        rg_of_row = np.empty(self.total_rows, dtype=np.int32)
        for rg_idx in range(self.num_row_groups):
            rg_start = self.rg_starts[rg_idx]
            rg_end = (
                self.rg_starts[rg_idx + 1] if rg_idx + 1 < self.num_row_groups else self.total_rows
            )
            rg_of_row[rg_start:rg_end] = rg_idx
        return rg_of_row

    def _decode_row_group(self, rg_idx: int):
        """
        Decode a full row group into a list of numpy arrays, one per row.
        Runs once per cache miss. Each row is a fresh contiguous numpy
        buffer, detached from the pyarrow chunked-array machinery, so
        subsequent reads are pure numpy slicing.
        """
        rg_tbl = self.pf.read_row_group(rg_idx, columns=["features", "seq_len", "n_features"])
        feats = rg_tbl["features"].combine_chunks()
        seq_lens = rg_tbl["seq_len"].to_numpy()
        n_feats = rg_tbl["n_features"].to_numpy()
        num_rows = len(seq_lens)

        decoded = []
        append = decoded.append
        for i in range(num_rows):
            buf = feats[i].as_buffer()
            arr = np.frombuffer(buf, dtype=np.float32).reshape(int(seq_lens[i]), int(n_feats[i]))
            append(np.ascontiguousarray(arr))
        return decoded

    def get_row_features(self, global_row_idx: int) -> np.ndarray:
        """Return the decoded feature window for a global row, filling the LRU cache on a miss."""
        rg_idx = bisect.bisect_right(self.rg_starts, global_row_idx) - 1
        rg_offset = global_row_idx - self.rg_starts[rg_idx]

        cached = self._cache.get(rg_idx)
        if cached is None:
            cached = self._decode_row_group(rg_idx)
            self._cache[rg_idx] = cached
            self._lru_order.append(rg_idx)
            if len(self._lru_order) > self.max_cached:
                evict = self._lru_order.pop(0)
                self._cache.pop(evict, None)
        else:
            self._lru_order.remove(rg_idx)
            self._lru_order.append(rg_idx)

        return cached[rg_offset]


def load_shard(
    shard_dir: str,
    chunk_idx: int | None = None,
    mmap_mode: Literal["c", "r", "r+", "w+"] | None = None,
) -> tuple:
    """
    Loads OHLCV indicator tensor, direction targets, magnitude targets, and timestamps.
    Supports both unchunked shards (X.npy) and chunked shards (X_chunk{c}.npy).
    """
    x_path, ydir_path, ymag_path, ts_path = shard_file_paths(shard_dir, chunk_idx)

    if not (os.path.exists(x_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return (None, None, None, None)
    try:
        x_data = np.load(x_path, mmap_mode=mmap_mode)
        y_dir = np.load(ydir_path, mmap_mode=mmap_mode)
        y_mag = np.load(ymag_path, mmap_mode=mmap_mode)
        timestamps = np.load(ts_path, mmap_mode=mmap_mode) if os.path.exists(ts_path) else None
        return (x_data, y_dir, y_mag, timestamps)
    except (OSError, ValueError, EOFError) as e:
        # np.load: OSError for unreadable files; ValueError/EOFError for corrupt/truncated arrays.
        logger.warning("Failed to load shard from %s (chunk: %s): %s", shard_dir, chunk_idx, e)
        print(f" [WARNING] Failed to load shard from {shard_dir} (chunk: {chunk_idx}): {e}")
        return (None, None, None, None)
