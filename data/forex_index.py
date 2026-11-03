"""Sample index construction for ``ForexDataset`` (Parquet years, year dirs, legacy folds)."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pyarrow.parquet as pq

from data.forex_shards import ParquetRowGroupCache, load_shard
from data.forex_spec import ALIAS_PAIRS, PAIR_INDEX

logger = logging.getLogger("lemtrain.data.forex_dataset")

_YEAR = re.compile(r"(20\d\d)")
_META_COLUMNS = [
    "pair",
    "timeframe",
    "timestamp",
    "y_dir",
    "tp_pips",
    "sl_pips",
    "seq_len",
    "n_features",
]
# Pair folders whose presence marks a Kaggle input directory as a Forex year directory.
_KAGGLE_PAIR_MARKERS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]


def _looks_like_forex(name: str) -> bool:
    """True for names that mention forex, a universe or a 20xx year."""
    return bool("forex" in name.lower() or "universe" in name.lower() or re.search(r"20\d\d", name))


class YearSources:
    """Year-keyed Parquet files and shard directories found under the shard roots."""

    def __init__(self) -> None:
        self.parquet: dict[str, str] = {}
        self.dirs: dict[str, str] = {}

    def register_parquet(self, path_str: str, base_name: str) -> None:
        """Register a Parquet year file under its stem, ``ForexUniverseYYYY`` and ``YYYY``."""
        self.parquet[base_name.replace(".parquet", "")] = path_str
        m = _YEAR.search(base_name)
        if m:
            yr = m.group(1)
            self.parquet[f"ForexUniverse{yr}"] = path_str
            self.parquet[yr] = path_str

    def register_dir(self, path_str: str, base_name: str) -> None:
        """Register a shard directory under its name, ``ForexUniverseYYYY`` and ``YYYY``."""
        self.dirs[base_name] = path_str
        m = _YEAR.search(base_name)
        if m:
            yr = m.group(1)
            self.dirs[f"ForexUniverse{yr}"] = path_str
            self.dirs[yr] = path_str

    def __len__(self) -> int:
        return len(self.parquet) + len(self.dirs)

    def has(self, name: str) -> bool:
        """True when ``name`` is a registered Parquet file or directory."""
        return name in self.parquet or name in self.dirs

    def scan_root(self, root: str) -> None:
        """Register a shard root itself and its Parquet files and subdirectories."""
        bname = os.path.basename(root)
        if _looks_like_forex(bname) and bname.endswith(".parquet") and os.path.isfile(root):
            self.register_parquet(root, bname)
        elif _looks_like_forex(bname) and os.path.isdir(root):
            self.register_dir(root, bname)
        try:
            for sd in os.listdir(root):
                full_p = os.path.join(root, sd)
                if sd.endswith(".parquet") and os.path.isfile(full_p):
                    self.register_parquet(full_p, sd)
                elif not sd.endswith(".zip") and os.path.isdir(full_p) and sd not in self.dirs:
                    self.register_dir(full_p, sd)
        except OSError as exc:
            logger.warning("Skipping unreadable forex root %s: %s", root, exc)

    def _scan_kaggle_dirs(self, root_dir: str, dirs: list[str]) -> None:
        for d in dirs:
            if not d.endswith(".zip") and _looks_like_forex(d):
                cand = os.path.join(root_dir, d)
                try:
                    if any(p in os.listdir(cand) for p in _KAGGLE_PAIR_MARKERS):
                        self.register_dir(cand, d)
                except OSError as exc:
                    logger.debug("Skipping unreadable Kaggle input dir %s: %s", cand, exc)

    def scan_kaggle_input(self) -> None:
        """Register Forex Parquet files and year directories under ``/kaggle/input``."""
        try:
            for root_dir, dirs, files in os.walk("/kaggle/input"):
                for f in files:
                    if f.endswith(".parquet") and _looks_like_forex(f):
                        self.register_parquet(os.path.join(root_dir, f), f)
                self._scan_kaggle_dirs(root_dir, dirs)
        except OSError as exc:
            logger.warning("Scan of /kaggle/input failed: %s", exc)

    def target_years(self, fold: int, is_train: bool) -> list[str]:
        """Return the walk-forward years of ``fold`` (or the years on disk as a fallback)."""
        fold_idx = max(1, min(6, fold))
        if is_train:
            candidate_years = [f"ForexUniverse{yr}" for yr in range(2019, 2019 + fold_idx + 1)]
        else:
            candidate_years = [f"ForexUniverse{2019 + fold_idx + 1}"]

        valid_targets = [y for y in candidate_years if self.has(y)]
        if valid_targets:
            return valid_targets
        # Dynamic fallback: Use whatever years are present on disk
        keys = list(self.parquet.keys()) + list(self.dirs.keys())
        available = sorted(
            {k for k in keys if k.startswith("ForexUniverse") and re.search(r"\d{4}", k)}
        )
        if not available:
            available = sorted(keys)
        if not available:
            return []
        if is_train:
            return available[:-1] if len(available) > 1 else available
        return [available[-1]]


class ForexIndexMixin:
    """Builds the flat ``(pair_idx, timeframe, key, row)`` sample index of ``ForexDataset``."""

    def resolve_pair_dir(self, parent_dir: str, pair: str) -> str | None:
        """Resolves pair directory handling symbol aliases (e.g. NAS100 <-> USTEC)."""
        candidates = [pair, ALIAS_PAIRS.get(pair, pair)]
        for k, v in ALIAS_PAIRS.items():
            if v == pair and k not in candidates:
                candidates.append(k)
        for cand in candidates:
            p = os.path.join(parent_dir, cand)
            if os.path.isdir(p):
                return p
        return None

    def build_index(self):
        """Build flat sample index over all pairs x active_timeframes x rows."""
        self._shard_paths = {}
        self._shards = {}
        self._index = []
        self._tf_map = {}

        sources = YearSources()
        for root in self.shard_roots:
            if os.path.exists(root):
                sources.scan_root(root)

        if os.path.exists("/kaggle/input") and len(sources) < 8:
            sources.scan_kaggle_input()

        if len(sources):
            for yr_name in sources.target_years(self.fold, self.is_train):
                if yr_name in sources.parquet:
                    self._index_parquet_year(yr_name, sources.parquet[yr_name])
                else:
                    self._index_year_dir(yr_name, sources.dirs.get(yr_name))
        else:
            self._index_legacy_folds()

        self.all_samples = list(self._index)

        if self.is_train and self.sample_fraction < 1.0:
            stride = max(1, round(1.0 / max(1e-4, self.sample_fraction)))
            self._index = self.all_samples[::stride]

        if len(self._index) == 0:
            print(
                f" [ForexDataset] NOTICE: No matching shards found in {self.shard_roots} "
                f"for pairs={self.pairs}, TFs={self.active_timeframes} (Fold: {self.fold})."
            )

    def _index_parquet_year(self, yr_name: str, p_path: str) -> None:
        """Index the matching rows of one Parquet year and its per-timeframe row lists."""
        cache = ParquetRowGroupCache(p_path)
        self._parquet_caches[yr_name] = cache
        meta_tbl = pq.read_table(p_path, columns=_META_COLUMNS)
        meta = {col: meta_tbl[col].to_numpy() for col in _META_COLUMNS if col != "pair"}
        meta = {"pair": meta_tbl["pair"].to_numpy(zero_copy_only=False), **meta}
        meta["rg_of_row"] = cache.row_group_of_rows()
        self._parquet_meta[yr_name] = meta

        pair_set = set(self.pairs)
        tf_set = set(self.active_timeframes)
        ts_arr = meta["timestamp"]
        for r, p_val in enumerate(meta["pair"]):
            p = str(p_val)
            tf = int(meta["timeframe"][r])
            if p in pair_set and tf in tf_set:
                self._index.append((PAIR_INDEX.get(p, 0), tf, ("parquet", yr_name), r))
                tf_key = (p, tf, yr_name)
                if tf_key not in self._parquet_tf_rows:
                    self._parquet_tf_rows[tf_key] = []
                    self._parquet_tf_ts[tf_key] = []
                self._parquet_tf_rows[tf_key].append(r)
                self._parquet_tf_ts[tf_key].append(ts_arr[r])

        for tf_key in list(self._parquet_tf_rows.keys()):
            if tf_key[2] == yr_name and isinstance(self._parquet_tf_rows[tf_key], list):
                self._parquet_tf_rows[tf_key] = np.asarray(
                    self._parquet_tf_rows[tf_key], dtype=np.int64
                )
                self._parquet_tf_ts[tf_key] = np.asarray(
                    self._parquet_tf_ts[tf_key], dtype=np.int64
                )

    def _index_year_dir(self, yr_name: str, yr_dir: str | None) -> None:
        """Index the ``<pair>/<tf>/X.npy`` or ``X_chunk*.npy`` shards of one year directory."""
        if not yr_dir or not os.path.isdir(yr_dir):
            return
        for pair in self.pairs:
            p_idx = PAIR_INDEX.get(pair, 0)
            pair_dir = self.resolve_pair_dir(yr_dir, pair)
            if not pair_dir:
                continue
            for tf in self.active_timeframes:
                tf_dir = os.path.join(pair_dir, str(tf))
                if os.path.isdir(tf_dir):
                    for c in self._chunk_ids(tf_dir):
                        self._index_chunk((pair, tf, yr_name, c), tf_dir, p_idx)

    @staticmethod
    def _chunk_ids(tf_dir: str) -> list[int]:
        """Return ``[-1]`` for an unchunked ``X.npy`` shard, else the sorted chunk numbers."""
        if os.path.exists(os.path.join(tf_dir, "X.npy")):
            return [-1]
        chunk_files = [
            f for f in os.listdir(tf_dir) if f.startswith("X_chunk") and f.endswith(".npy")
        ]
        return sorted(int(f.replace("X_chunk", "").replace(".npy", "")) for f in chunk_files)

    def _index_chunk(self, key: tuple, tf_dir: str, p_idx: int) -> None:
        """Register one shard (``key[3]`` is the chunk, -1 unchunked) and index its rows."""
        pair, tf, yr_name, c = key
        self._shard_paths[key] = (tf_dir, c)
        tf_group_key = (pair, tf, yr_name)
        if tf_group_key not in self._tf_map:
            self._tf_map[tf_group_key] = []
        self._tf_map[tf_group_key].append(key)

        x_data, _, _, _ = load_shard(tf_dir, chunk_idx=None if c < 0 else c, mmap_mode="r")
        if x_data is not None:
            length = len(x_data)
            del x_data
            for row in range(length):
                self._index.append((p_idx, tf, key, row))

    def _index_legacy_folds(self) -> None:
        """Index fold-sharded ``<pair>/<tf>/folds/<fold_N|val>`` shards."""
        for pair in self.pairs:
            p_idx = PAIR_INDEX.get(pair, 0)
            pair_root = next(
                (root for root in self.shard_roots if self.resolve_pair_dir(root, pair)), None
            )
            if pair_root is None:
                continue

            for tf in self.active_timeframes:
                if self.split == "val":
                    fold_names = ["val"]
                else:
                    max_fold = self.fold if self.fold is not None else 6
                    fold_names = [f"fold_{f}" for f in range(1, max_fold + 1)]
                for fold_name in fold_names:
                    shard_dir = os.path.join(pair_root, pair, str(tf), "folds", fold_name)
                    key = (pair, tf, fold_name, -1)
                    x_data, _, _, _ = load_shard(shard_dir, mmap_mode="r")
                    if x_data is not None:
                        self._shard_paths[key] = (shard_dir, -1)
                        length = len(x_data)
                        del x_data
                        for row in range(length):
                            self._index.append((p_idx, tf, key, row))
