"""
LemGendary RowGroupAwareSampler
================================
2026: Parquet-optimized training sampler.

Why this exists
---------------
Parquet is a columnar format with row-group granularity. Reading row N in
a row group of size G requires decompressing all G rows. Under `shuffle=True`,
PyTorch requests random rows, causing ~98% cache miss rate against the LRU
in ParquetRowGroupCache (12 cached groups / 700 total groups ~= 1.7% hit).

This sampler shuffles at the ROW GROUP level instead: it groups sample
indices by their parent row group, shuffles the group order, then yields
indices sequentially within each group. Result: after the first access to
a group, the next G-1 accesses are all cache hits. Hit rate climbs from
~2% to ~99%.

Compatible with
---------------
- Parquet-backed datasets (via `_parquet_meta[year]['rg_of_row']`)
- Legacy .npy shard datasets (each shard treated as one group)

Drop-in usage
-------------
    sampler = RowGroupAwareSampler(dataset, shuffle=True, seed=42)
    loader = DataLoader(dataset, batch_size=..., sampler=sampler, ...)

    # At the start of each epoch:
    sampler.set_epoch(epoch)
"""

import numpy as np
from torch.utils.data import Sampler


class RowGroupAwareSampler(Sampler):
    """
    Shuffles at the row-group level, sequential within a group.
    """

    def __init__(self, dataset, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Bucket sample indices by their locality key.
        #   Parquet:     ('parquet', year, row_group_id)
        #   Legacy .npy: ('shard', pair, tf, year_ident, chunk_idx)
        self._groups = {}
        self._flat_order = []

        index = getattr(dataset, '_index', [])
        for i, entry in enumerate(index):
            try:
                _p_idx, _tf, key, row = entry
            except (ValueError, TypeError):
                self._flat_order.append(i)
                continue

            locality_key = self._locality_key(key, row)
            self._groups.setdefault(locality_key, []).append(i)
            self._flat_order.append(i)

        self._group_keys = list(self._groups.keys())
        self._num_samples = len(self._flat_order)

    def _locality_key(self, key, row):
        """Derive a cache-locality key for a dataset entry."""
        # Parquet branch: key is ('parquet', year_name)
        if isinstance(key, tuple) and len(key) == 2 and key[0] == 'parquet':
            yr_name = key[1]
            meta = getattr(self.dataset, '_parquet_meta', {}).get(yr_name, {})
            rg_of_row = meta.get('rg_of_row')
            if rg_of_row is not None and 0 <= row < len(rg_of_row):
                return ('parquet', yr_name, int(rg_of_row[row]))
            return ('parquet', yr_name, -1)

        # Legacy shard branch: key is (pair, tf, year_ident, chunk_idx)
        if isinstance(key, tuple):
            return ('shard',) + tuple(str(x) for x in key)

        # Fallback
        return ('unknown', str(key))

    def set_epoch(self, epoch: int):
        """Reseed shuffling for a new epoch."""
        self.epoch = epoch

    def __iter__(self):
        if not self.shuffle:
            yield from self._flat_order
            return

        rng = np.random.default_rng(self.seed + self.epoch)

        keys = list(self._group_keys)
        rng.shuffle(keys)

        for k in keys:
            idxs = self._groups[k]
            if len(idxs) > 1:
                idxs = list(rng.permutation(idxs))
            yield from idxs

    def __len__(self):
        return self._num_samples