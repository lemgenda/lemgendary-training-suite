"""
LemGendary ForexDataset v3.1
==============================
PyTorch Dataset class supporting both year-based Walk-Forward manifolds
(ForexUniverse2019..2026) and legacy fold-sharded manifolds, with
chunked .npy loading, cross-timeframe alignment, and Governor integration.

Integrates with MultiTaskDataset pattern via task_type = "forex". Shard roots come from
``data.forex_roots``, the sample index from ``data.forex_index`` and the shard readers from
``data.forex_shards``.

v3.1 Changes:
  - Reverted the memmap feature cache (v3.0) back to the process-safe LRU
    row-group cache (v2.3). The memmap cache needs ~40 GB for fold 1's
    two-year training set, which does not fit in Kaggle's 20 GB disk or
    14 GB /dev/shm. The LRU cache peaks at ~1.7 GB RAM and works everywhere.

v2.3 Changes (retained):
  - ParquetRowGroupCache decodes row groups to per-row numpy arrays at
    cache-fill time. On a cache hit, get_row_features becomes a numpy slice
    instead of a pyarrow scalar extraction.

v2.2 Changes (retained):
  - Exposes _parquet_meta[year]['rg_of_row'] so RowGroupAwareSampler can
    shuffle at the row-group level (raises LRU hit rate from ~2% to ~99%).

v2.1 Changes (retained):
  - O(1) aligned-row lookup via precomputed alignment maps.
  - Eliminated double-copy on the legacy .npy path.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from data.forex_index import ForexIndexMixin, ForexSelection, LegacyShardStore, ParquetStore
from data.forex_roots import discover_shard_roots
from data.forex_shards import load_shard, safe_from_numpy
from data.forex_spec import EXTENDED_PAIRS, PAIR_PIP_SCALE, TIMEFRAME_LOOKBACK

logger = logging.getLogger("lemtrain.data.forex_dataset")

DEFAULT_TIMEFRAMES = [1, 5, 15, 60, 240, 1440]


@dataclass(frozen=True)
class ForexSplit:
    """Which samples a ``ForexDataset`` serves: pairs, timeframes, split, fraction and fold."""

    pairs: list | None = None
    active_timeframes: list | None = None
    is_train: bool = True
    sample_fraction: float = 1.0
    fold: int | None = None
    spread_stress_pips: float = 0.0


class ForexDataset(ForexIndexMixin, Dataset):
    """
    Multi-pair, multi-timeframe Forex Dataset for the LemGendary Training Suite.

    Loads windowed OHLCV + indicator samples from pre-built .npy shards.
    Supports year-based Walk-Forward folds (2019..2026), Governor-aligned
    fractional sampling, and multi-scale timeframe expansion.
    """

    task_type = "forex"
    size = None

    def __init__(self, shard_root: str, split: ForexSplit | None = None):
        split = split or ForexSplit()
        self.shard_roots = discover_shard_roots(shard_root)
        self.selection = ForexSelection(
            pairs=split.pairs if split.pairs else EXTENDED_PAIRS,
            active_timeframes=split.active_timeframes or list(DEFAULT_TIMEFRAMES),
            is_train=split.is_train,
            sample_fraction=split.sample_fraction,
            split="train" if split.is_train else "val",
            fold=split.fold if split.fold is not None else 1,
            spread_stress_pips=split.spread_stress_pips,
        )
        self.parquet_store = ParquetStore()
        self.legacy_shards = LegacyShardStore()
        self._index = []
        self.all_samples = []

        self.build_index()

    def update_strategy(
        self,
        fraction: float | None = None,
        active_timeframes: list | None = None,
        size: float | tuple | None = None,
        stress: float | None = None,
    ):
        """
        Governor hook: update sampling fraction or active timeframes mid-training.
        Mirrors MultiTaskDataset.update_strategy() interface; ``size`` is accepted for
        that interface but ignored because forex windows have fixed per-timeframe lookbacks.
        """
        if size is not None:
            logger.debug("ForexDataset ignores size=%s (fixed per-timeframe lookbacks)", size)
        rebuild = False
        if active_timeframes is not None and active_timeframes != self.selection.active_timeframes:
            self.selection.active_timeframes = active_timeframes
            rebuild = True
        if stress is not None:
            self.selection.spread_stress_pips = stress

        if fraction is not None and fraction != self.selection.sample_fraction:
            self.selection.sample_fraction = fraction
            if not rebuild and hasattr(self, "all_samples") and self.all_samples:
                stride = (
                    max(1, round(1.0 / max(1e-4, self.selection.sample_fraction)))
                    if (self.selection.is_train and self.selection.sample_fraction < 1.0)
                    else 1
                )
                self._index = self.all_samples[::stride]
                return

        if rebuild:
            self.build_index()

    def __len__(self) -> int:
        return max(1, len(self._index))

    def _get_shard_data(self, key: tuple):
        """Lazily retrieves or loads shard arrays for a specific key."""
        if key not in self.legacy_shards.shards:
            tf_dir, chunk_idx = self.legacy_shards.shard_paths[key]
            c_arg = chunk_idx if chunk_idx >= 0 else None
            self.legacy_shards.shards[key] = load_shard(tf_dir, chunk_idx=c_arg, mmap_mode="r")
        return self.legacy_shards.shards[key]

    def _get_alignment_map(self, primary_key: tuple, other_key: tuple):
        """Lazily computes or retrieves precalculated alignment row mapping between two shards."""
        pair_key = (primary_key, other_key)
        if pair_key in self.legacy_shards.alignment_cache:
            return self.legacy_shards.alignment_cache[pair_key]

        _, _, _, prim_ts = self._get_shard_data(primary_key)
        other_x, _, _, other_ts = self._get_shard_data(other_key)

        if other_x is None or len(other_x) == 0 or prim_ts is None or len(prim_ts) == 0:
            self.legacy_shards.alignment_cache[pair_key] = None
            return None

        if (
            other_ts is not None
            and len(other_ts) > 1
            and len(prim_ts) > 1
            and prim_ts[-1] > prim_ts[0]
            and other_ts[-1] > other_ts[0]
        ):
            aligned = np.searchsorted(other_ts, prim_ts, side="right") - 1
            aligned = np.clip(aligned, 0, len(other_x) - 1).astype(np.int64)
        else:
            prim_tf = primary_key[1]
            other_tf = other_key[1]
            ratio = prim_tf / other_tf
            rows = np.arange(len(prim_ts), dtype=np.float64)
            aligned = np.clip((rows * ratio).astype(np.int64), 0, len(other_x) - 1)

        self.legacy_shards.alignment_cache[pair_key] = aligned
        return aligned

    def _get_parquet_alignment(self, pair_name: str, primary_tf: int, other_tf: int, yr_name: str):
        align_key = (pair_name, primary_tf, other_tf, yr_name)
        if align_key in self.parquet_store.alignment_cache:
            return self.parquet_store.alignment_cache[align_key]

        other_rows = self.parquet_store.tf_rows.get((pair_name, other_tf, yr_name))
        other_ts = self.parquet_store.tf_ts.get((pair_name, other_tf, yr_name))
        prim_ts = self.parquet_store.tf_ts.get((pair_name, primary_tf, yr_name))

        if (
            other_rows is None
            or other_ts is None
            or prim_ts is None
            or any(len(a) == 0 for a in (other_rows, other_ts, prim_ts))
        ):
            self.parquet_store.alignment_cache[align_key] = None
            return None

        mapped = np.searchsorted(other_ts, prim_ts, side="right") - 1
        mapped = np.clip(mapped, 0, len(other_rows) - 1).astype(np.int64)
        result = (other_rows, mapped)
        self.parquet_store.alignment_cache[align_key] = result
        return result

    @staticmethod
    def _fit_window(raw_sample: np.ndarray, target_len: int) -> np.ndarray:
        """
        Left-pad or right-crop a window to target_len along the time axis.
        Returns a numpy array (potentially a view when no padding is needed).
        """
        cur = raw_sample.shape[0]
        if cur < target_len:
            pad = np.zeros((target_len - cur, raw_sample.shape[1]), dtype=raw_sample.dtype)
            return np.concatenate([pad, raw_sample], axis=0)
        if cur > target_len:
            return raw_sample[-target_len:]
        return raw_sample

    def _sample(self, pair_name: str, direction, tp_sl: tuple, p_idx: int) -> tuple:
        """Build the labels (spread-stressed magnitudes in pair pip units) of one sample."""
        scale = PAIR_PIP_SCALE.get(pair_name, 1.0)
        mag_tp = float(tp_sl[0]) / scale
        mag_sl = float(tp_sl[1]) / scale
        if self.selection.spread_stress_pips > 0.0:
            mag_tp = max(0.0, mag_tp - (self.selection.spread_stress_pips / scale))
            mag_sl = mag_sl + (self.selection.spread_stress_pips / scale)

        labels = {
            "direction": torch.tensor(int(direction), dtype=torch.long),
            "magnitude": torch.tensor([mag_tp, mag_sl], dtype=torch.float32),
        }
        return labels, torch.tensor(p_idx, dtype=torch.long)

    def _parquet_aligned_row(self, primary: tuple, other_tf: int) -> int | None:
        """Return the ``other_tf`` row aligned to a Parquet row, or ``None`` when unaligned."""
        pair_name, tf, yr_name, row = primary
        align = self._get_parquet_alignment(pair_name, tf, other_tf, yr_name)
        if align is None:
            return None

        other_rows, mapped = align
        prim_rows = self.parquet_store.tf_rows.get((pair_name, tf, yr_name))
        if prim_rows is None or len(prim_rows) == 0:
            return None

        pos = bisect.bisect_left(prim_rows, row)
        if pos >= len(prim_rows) or int(prim_rows[pos]) != int(row):
            return None
        return int(other_rows[int(mapped[pos])])

    def _parquet_other_window(self, primary: tuple, other_tf: int, n_features: int):
        """Return the window of ``other_tf`` aligned to a Parquet row (zeros when unaligned)."""
        other_target_len = TIMEFRAME_LOOKBACK.get(other_tf, 168)
        aligned_row = self._parquet_aligned_row(primary, other_tf)
        if aligned_row is None:
            return torch.zeros(other_target_len, n_features)
        cache = self.parquet_store.caches[primary[2]]
        other_sample = self._fit_window(cache.get_row_features(aligned_row), other_target_len)
        return safe_from_numpy(other_sample)

    def _parquet_item(self, p_idx: int, tf: int, yr_name: str, row: int):
        """Native high-throughput Parquet retrieval of one sample."""
        meta = self.parquet_store.meta[yr_name]
        cache = self.parquet_store.caches[yr_name]
        pair_name = str(meta["pair"][row])

        target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
        raw_sample = self._fit_window(cache.get_row_features(row), target_len)

        # Single-copy tensor materialization.
        tf_inputs = {tf: safe_from_numpy(raw_sample)}
        primary = (pair_name, tf, yr_name, row)
        for other_tf in self.selection.active_timeframes:
            if other_tf != tf:
                tf_inputs[other_tf] = self._parquet_other_window(
                    primary, other_tf, raw_sample.shape[1]
                )

        labels, pair_idx = self._sample(
            pair_name, meta["y_dir"][row], (meta["tp_pips"][row], meta["sl_pips"][row]), p_idx
        )
        return (tf_inputs, labels, pair_idx)

    def _legacy_other_window(
        self, key: tuple[str, int, str, int], row: int, other_tf: int, n_features: int
    ):
        """Return the window of ``other_tf`` aligned to a legacy shard row (zeros if absent)."""
        other_target_len = TIMEFRAME_LOOKBACK.get(other_tf, 168)
        yr_identifier = str(key[2]) if len(key) > 2 else ""
        candidate_keys = self.legacy_shards.tf_map.get((str(key[0]), other_tf, yr_identifier), [])
        if not candidate_keys:
            return torch.zeros(other_target_len, n_features)

        other_key = candidate_keys[0]
        other_x, _, _, _ = self._get_shard_data(other_key)
        if other_x is None or len(other_x) == 0:
            return torch.zeros(other_target_len, n_features)

        alignment_map = self._get_alignment_map(key, other_key)
        if alignment_map is not None and row < len(alignment_map):
            aligned_row = int(alignment_map[row])
        else:
            ratio = key[1] / other_tf
            aligned_row = min(int(row * ratio), len(other_x) - 1)

        other_sample = self._fit_window(np.asarray(other_x[aligned_row]), other_target_len)
        return safe_from_numpy(other_sample)

    def _legacy_item(self, p_idx: int, tf: int, key: tuple, row: int):
        """Legacy ``.npy`` shard retrieval of one sample."""
        x_data, y_dir, y_mag, _ = self._get_shard_data(key)

        target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
        raw_sample = self._fit_window(np.asarray(x_data[row]), target_len)

        tf_inputs = {tf: safe_from_numpy(raw_sample)}
        for other_tf in self.selection.active_timeframes:
            if other_tf != tf:
                tf_inputs[other_tf] = self._legacy_other_window(
                    key, row, other_tf, raw_sample.shape[1]
                )

        labels, pair_idx = self._sample(
            str(key[0]), y_dir[row], (y_mag[row, 0], y_mag[row, 1]), p_idx
        )
        return (tf_inputs, labels, pair_idx)

    def __getitem__(self, index: int):
        """
        Returns:
            tf_inputs: Dict[int -> Tensor[seq_len, features]]
            labels:    Dict with 'direction' (long) and 'magnitude' (float32 [2])
            pair_idx:  Long tensor (scalar)
        """
        if len(self._index) == 0:
            tf_inputs = {
                tf: torch.zeros(TIMEFRAME_LOOKBACK.get(tf, 168), 14)
                for tf in self.selection.active_timeframes
            }
            return (
                tf_inputs,
                {
                    "direction": torch.tensor(1, dtype=torch.long),
                    "magnitude": torch.zeros(2, dtype=torch.float32),
                },
                torch.tensor(0, dtype=torch.long),
            )

        p_idx, tf, key, row = self._index[index]
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "parquet":
            return self._parquet_item(p_idx, tf, key[1], row)
        return self._legacy_item(p_idx, tf, key, row)
