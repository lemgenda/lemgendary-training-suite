"""
LemGendary ForexDataset v2.0
==============================
PyTorch Dataset class supporting both year-based Walk-Forward manifolds
(ForexUniverse2019..2026) and legacy fold-sharded manifolds, with
chunked .npy loading, cross-timeframe alignment, and Governor integration.

Integrates with MultiTaskDataset pattern via task_type = "forex".
"""

import os
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset

TIMEFRAME_RUNGS = [1, 5, 15, 60, 240, 1440]
TIMEFRAME_LOOKBACK = {
    1: 512,
    5: 288,
    15: 192,
    60: 168,
    240: 90,
    1440: 252,
}

MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
EXTENDED_PAIRS = [
    # G7 Majors (4 Core + 4 G7)
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    # High-Beta Crosses
    "EURJPY", "GBPJPY", "EURGBP",
    # Commodities & Energy
    "XAGUSD", "USOIL",
    # Global Equity Indices
    "US500", "NAS100", "DE40"
]

# Alias mapping for alternate broker tickers
ALIAS_PAIRS = {
    "USTEC": "NAS100",
    "GER40": "DE40",
}

PAIR_INDEX = {p: i for i, p in enumerate(EXTENDED_PAIRS)}
PAIR_INDEX["USTEC"] = PAIR_INDEX["NAS100"]
PAIR_INDEX["GER40"] = PAIR_INDEX["DE40"]

PAIR_PIP_SCALE = {
    "EURUSD": 1.0, "GBPUSD": 1.0, "USDJPY": 1.0, "USDCAD": 1.0,
    "USDCHF": 1.0, "AUDUSD": 1.0, "NZDUSD": 1.0,
    "EURGBP": 1.0, "EURJPY": 1.0, "GBPJPY": 1.0,
    "XAGUSD": 5.0, "USOIL": 5.0, "XAUUSD": 10.0,
    "US500": 20.0, "DE40": 20.0, "GER40": 20.0,
    "NAS100": 40.0, "USTEC": 40.0
}


class ParquetRowGroupCache:  # pylint: disable=too-few-public-methods
    """
    Process-safe LRU cache for Parquet row groups.
    Caches unpacked binary float buffers to deliver sub-microsecond row access.
    """
    def __init__(self, parquet_path: str, max_cached_groups: int = 4):
        import pyarrow.parquet as pq
        self.parquet_path = parquet_path
        self.max_cached = max_cached_groups
        self.pf = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.pf.num_row_groups
        self.metadata = self.pf.metadata
        self._cache = {}
        self._lru_order = []

        self.rg_starts = []
        curr = 0
        for i in range(self.num_row_groups):
            self.rg_starts.append(curr)
            curr += self.metadata.row_group(i).num_rows
        self.total_rows = curr

    def get_row_features(self, global_row_idx: int) -> np.ndarray:
        import bisect
        rg_idx = bisect.bisect_right(self.rg_starts, global_row_idx) - 1
        rg_offset = global_row_idx - self.rg_starts[rg_idx]

        if rg_idx not in self._cache:
            rg_tbl = self.pf.read_row_group(rg_idx, columns=['features', 'seq_len', 'n_features'])
            feats = rg_tbl['features']
            seq_lens = rg_tbl['seq_len'].to_numpy()
            n_feats = rg_tbl['n_features'].to_numpy()
            self._cache[rg_idx] = (seq_lens, n_feats, feats)
            self._lru_order.append(rg_idx)

            if len(self._lru_order) > self.max_cached:
                evict = self._lru_order.pop(0)
                del self._cache[evict]
        else:
            self._lru_order.remove(rg_idx)
            self._lru_order.append(rg_idx)

        seq_lens, n_feats, feats = self._cache[rg_idx]
        buf = feats[rg_offset].as_buffer()
        s_len = int(seq_lens[rg_offset])
        n_feat = int(n_feats[rg_offset])
        return np.frombuffer(buf, dtype=np.float32).reshape(s_len, n_feat)


def load_shard(
    shard_dir: str,
    chunk_idx: int | None = None,
    mmap_mode: Literal['c', 'r', 'r+', 'w+'] | None = None
) -> tuple:
    """
    Loads OHLCV indicator tensor, direction targets, magnitude targets, and timestamps.
    Supports both unchunked shards (X.npy) and chunked shards (X_chunk{c}.npy).
    """
    if chunk_idx is not None and chunk_idx >= 0:
        X_path = os.path.join(shard_dir, f'X_chunk{chunk_idx}.npy')
        ydir_path = os.path.join(shard_dir, f'y_dir_chunk{chunk_idx}.npy')
        ymag_path = os.path.join(shard_dir, f'y_mag_chunk{chunk_idx}.npy')
        ts_path = os.path.join(shard_dir, f'timestamps_chunk{chunk_idx}.npy')
    else:
        X_path = os.path.join(shard_dir, 'X.npy')
        ydir_path = os.path.join(shard_dir, 'y_dir.npy')
        ymag_path = os.path.join(shard_dir, 'y_mag.npy')
        ts_path = os.path.join(shard_dir, 'timestamps.npy')

    if not (os.path.exists(X_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return (None, None, None, None)
    try:
        X = np.load(X_path, mmap_mode=mmap_mode)
        y_dir = np.load(ydir_path, mmap_mode=mmap_mode)
        y_mag = np.load(ymag_path, mmap_mode=mmap_mode)
        timestamps = np.load(ts_path, mmap_mode=mmap_mode) if os.path.exists(ts_path) else None
        return (X, y_dir, y_mag, timestamps)
    except Exception as e:
        print(f" [WARNING] Failed to load shard from {shard_dir} (chunk: {chunk_idx}): {e}")
        return (None, None, None, None)


class ForexDataset(Dataset):
    """
    Multi-pair, multi-timeframe Forex Dataset for the LemGendary Training Suite.

    Loads windowed OHLCV + indicator samples from pre-built .npy shards.
    Supports year-based Walk-Forward folds (2019..2026), Governor-aligned
    fractional sampling, and multi-scale timeframe expansion.

    Args:
        shard_root:         Root directory containing shards.
        pairs:              List of currency pair symbols to include.
        active_timeframes:  List of active timeframe rungs (minutes).
        is_train:           True for training split, False for validation.
        sample_fraction:    Fraction of training samples to use (Governor managed).
        fold:               Walk-forward fold index (1..6).
        spread_stress_pips: Dynamic spread friction in pips.
    """

    task_type = "forex"
    size = None

    def __init__(
        self,
        shard_root: str,
        pairs: list | None = None,
        active_timeframes: list | None = None,
        is_train: bool = True,
        sample_fraction: float = 1.0,
        fold: int | None = None,
        spread_stress_pips: float = 0.0,
    ):
        import yaml
        unified_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unified_models_v2.yaml")
        gdrive_ids = []
        if os.path.exists(unified_path):
            try:
                with open(unified_path, 'r', encoding="utf-8") as f:
                    um = yaml.safe_load(f)
                    gdrive_ids = um.get("forex_predictor", {}).get("google_drive_dataset_ids", [])
            except Exception:
                pass

        env = os.environ.get("ENV", "local")
        import sys
        if 'colab' in sys.modules or os.path.exists('/content'):
            env = 'colab'
        elif os.path.exists('/kaggle'):
            env = 'kaggle'

        resolved_roots = []
        if shard_root and os.path.exists(shard_root):
            resolved_roots.append(os.path.abspath(shard_root))

        if env == 'colab':
            base_drive = "/content/drive/MyDrive/LemGendaryDatasets"
            if os.path.exists(base_drive):
                for d in os.listdir(base_drive):
                    if "forex" in d.lower():
                        cand_forex = os.path.join(base_drive, d, "forex")
                        cand = cand_forex if os.path.isdir(cand_forex) else os.path.join(base_drive, d)
                        if os.path.isdir(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)
            if not resolved_roots:
                print(f"\n[ERROR] Colab requires Google Drive dataset packages at {base_drive}")
                sys.exit(1)

        elif env == 'kaggle':
            if os.path.exists('/kaggle/input'):
                for root_dir, dirs, _ in os.walk('/kaggle/input'):
                    if 'forex' in dirs:
                        p = os.path.join(root_dir, 'forex')
                        if p not in resolved_roots:
                            resolved_roots.append(p)
                    for d in dirs:
                        if "forex" in d.lower():
                            p = os.path.join(root_dir, d)
                            if p not in resolved_roots:
                                resolved_roots.append(p)
            if not resolved_roots:
                print("\n[WARNING] No attached forex datasets found in Kaggle /kaggle/input!")

        else:  # Local
            base_search = [
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets")),
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
            ]
            for bs in base_search:
                if not os.path.exists(bs):
                    continue
                try:
                    entries = os.listdir(bs)
                except OSError:
                    continue
                for d in entries:
                    if "forex" in d.lower():
                        cand_forex = os.path.join(bs, d, "forex")
                        cand = cand_forex if os.path.isdir(cand_forex) else os.path.join(bs, d)
                        if os.path.isdir(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)

            if not resolved_roots and gdrive_ids:
                print(f"\n[DATA] Local forex dataset not found. Attempting gdown from {len(gdrive_ids)} Google Drive IDs...")
                import subprocess
                import zipfile
                base_out = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets"))
                os.makedirs(base_out, exist_ok=True)
                for gid in gdrive_ids:
                    dest = os.path.join(base_out, f"dataset_{gid}.zip")
                    subprocess.run([sys.executable, "-m", "gdown", "--id", gid, "-O", dest], check=False)
                    if os.path.exists(dest):
                        with zipfile.ZipFile(dest, 'r') as zip_ref:
                            zip_ref.extractall(base_out)
                        os.remove(dest)
                for d in os.listdir(base_out):
                    if "forex" in d.lower():
                        cand = os.path.join(base_out, d, "forex") if os.path.isdir(os.path.join(base_out, d, "forex")) else os.path.join(base_out, d)
                        if os.path.exists(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)

        self.shard_roots = resolved_roots
        if pairs:
            self.pairs = pairs
        else:
            self.pairs = EXTENDED_PAIRS

        self.active_timeframes = active_timeframes or [1, 5, 15, 60, 240, 1440]
        self.is_train = is_train
        self.sample_fraction = sample_fraction
        self.split = "train" if is_train else "val"
        self.fold = fold if fold is not None else 1
        self.spread_stress_pips = spread_stress_pips

        self._parquet_caches = {}
        self._parquet_meta = {}
        self._parquet_tf_rows = {}
        self._parquet_tf_ts = {}

        self._build_index()

    def _resolve_pair_dir(self, parent_dir: str, pair: str) -> str | None:
        """Resolves pair directory handling symbol aliases (e.g. NAS100 <-> USTEC)."""
        candidates = [pair, ALIAS_PAIRS.get(pair, pair)]
        # Also check reverse alias
        for k, v in ALIAS_PAIRS.items():
            if v == pair and k not in candidates:
                candidates.append(k)
        for cand in candidates:
            p = os.path.join(parent_dir, cand)
            if os.path.isdir(p):
                return p
        return None

    def _build_index(self):
        """Build flat sample index over all pairs x active_timeframes x rows."""
        self._shard_paths = {}
        self._shards = {}
        self._index = []
        self._tf_map = {}

        # 1. Detect if any attached root has year-based structure (ForexUniverseYYYY.parquet or ForexUniverseYYYY/)
        # Supports single unified folder or multiple distinct dataset roots (e.g. multi-dataset mounts on Kaggle)
        year_parquet_map = {}
        year_dirs_map = {}
        for root in self.shard_roots:
            if not os.path.exists(root):
                continue
            bname = os.path.basename(root)
            if bname.startswith("ForexUniverse") and bname.endswith(".parquet") and os.path.isfile(root):
                year_parquet_map[bname.replace(".parquet", "")] = root
            elif bname.startswith("ForexUniverse") and os.path.isdir(root):
                year_dirs_map[bname] = root
            try:
                for sd in os.listdir(root):
                    full_p = os.path.join(root, sd)
                    if sd.startswith("ForexUniverse") and sd.endswith(".parquet") and os.path.isfile(full_p):
                        year_parquet_map[sd.replace(".parquet", "")] = full_p
                    elif sd.startswith("ForexUniverse") and not sd.endswith(".zip") and os.path.isdir(full_p):
                        if sd not in year_dirs_map:
                            year_dirs_map[sd] = full_p
            except OSError:
                continue

        # In Kaggle environment, scan /kaggle/input
        if os.path.exists('/kaggle/input') and (len(year_parquet_map) + len(year_dirs_map)) < 8:
            try:
                for root_dir, dirs, files in os.walk('/kaggle/input'):
                    for f in files:
                        if f.startswith("ForexUniverse") and f.endswith(".parquet"):
                            yk = f.replace(".parquet", "")
                            if yk not in year_parquet_map:
                                year_parquet_map[yk] = os.path.join(root_dir, f)
                    for d in dirs:
                        if d.startswith("ForexUniverse") and not d.endswith(".zip"):
                            cand = os.path.join(root_dir, d)
                            if d not in year_dirs_map:
                                try:
                                    if any(p in os.listdir(cand) for p in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']):
                                        year_dirs_map[d] = cand
                                except OSError:
                                    pass
            except OSError:
                pass

        if year_parquet_map or year_dirs_map:
            # --- Year-Based Walk-Forward Manifold ---
            # Fold k: Train = [2019..2019+k], Val = [2019+k+1]
            fold_idx = max(1, min(6, self.fold))
            if self.is_train:
                target_years = [f"ForexUniverse{yr}" for yr in range(2019, 2019 + fold_idx + 1)]
            else:
                target_years = [f"ForexUniverse{2019 + fold_idx + 1}"]

            for yr_name in target_years:
                if yr_name in year_parquet_map:
                    p_path = year_parquet_map[yr_name]
                    cache = ParquetRowGroupCache(p_path)
                    self._parquet_caches[yr_name] = cache

                    import pyarrow.parquet as pq
                    meta_tbl = pq.read_table(
                        p_path,
                        columns=['pair', 'timeframe', 'timestamp', 'y_dir', 'tp_pips', 'sl_pips', 'seq_len', 'n_features']
                    )
                    pairs_arr = meta_tbl['pair'].to_numpy(zero_copy_only=False)
                    tfs_arr = meta_tbl['timeframe'].to_numpy()
                    ts_arr = meta_tbl['timestamp'].to_numpy()
                    ydir_arr = meta_tbl['y_dir'].to_numpy()
                    tp_arr = meta_tbl['tp_pips'].to_numpy()
                    sl_arr = meta_tbl['sl_pips'].to_numpy()
                    seqlen_arr = meta_tbl['seq_len'].to_numpy()
                    nfeat_arr = meta_tbl['n_features'].to_numpy()

                    self._parquet_meta[yr_name] = {
                        "pair": pairs_arr,
                        "timeframe": tfs_arr,
                        "timestamp": ts_arr,
                        "y_dir": ydir_arr,
                        "tp_pips": tp_arr,
                        "sl_pips": sl_arr,
                        "seq_len": seqlen_arr,
                        "n_features": nfeat_arr,
                    }

                    pair_set = set(self.pairs)
                    tf_set = set(self.active_timeframes)

                    for r, p_val in enumerate(pairs_arr):
                        p = str(p_val)
                        tf = int(tfs_arr[r])
                        if p in pair_set and tf in tf_set:
                            p_idx = PAIR_INDEX.get(p, 0)
                            self._index.append((p_idx, tf, ("parquet", yr_name), r))
                            tf_key = (p, tf, yr_name)
                            if tf_key not in self._parquet_tf_rows:
                                self._parquet_tf_rows[tf_key] = []
                                self._parquet_tf_ts[tf_key] = []
                            self._parquet_tf_rows[tf_key].append(r)
                            self._parquet_tf_ts[tf_key].append(ts_arr[r])

                    for tf_key in list(self._parquet_tf_rows.keys()):
                        if tf_key[2] == yr_name and isinstance(self._parquet_tf_rows[tf_key], list):
                            self._parquet_tf_rows[tf_key] = np.asarray(self._parquet_tf_rows[tf_key], dtype=np.int64)
                            self._parquet_tf_ts[tf_key] = np.asarray(self._parquet_tf_ts[tf_key], dtype=np.int64)

                elif yr_name in year_dirs_map:
                    yr_dir = year_dirs_map.get(yr_name)
                    if not yr_dir or not os.path.isdir(yr_dir):
                        continue

                    for pair in self.pairs:
                        p_idx = PAIR_INDEX.get(pair, 0)
                        pair_dir = self._resolve_pair_dir(yr_dir, pair)
                        if not pair_dir:
                            continue

                        for tf in self.active_timeframes:
                            tf_dir = os.path.join(pair_dir, str(tf))
                            if not os.path.isdir(tf_dir):
                                continue

                            X_path = os.path.join(tf_dir, "X.npy")
                            if os.path.exists(X_path):
                                key = (pair, tf, yr_name, -1)
                                self._shard_paths[key] = (tf_dir, -1)
                                tf_group_key = (pair, tf, yr_name)
                                if tf_group_key not in self._tf_map:
                                    self._tf_map[tf_group_key] = []
                                self._tf_map[tf_group_key].append(key)

                                X, _, _, _ = load_shard(tf_dir, chunk_idx=None, mmap_mode="r")
                                if X is not None:
                                    length = len(X)
                                    del X
                                    for row in range(length):
                                        self._index.append((p_idx, tf, key, row))
                            else:
                                chunk_files = [f for f in os.listdir(tf_dir) if f.startswith("X_chunk") and f.endswith(".npy")]
                                if chunk_files:
                                    chunk_indices = sorted([int(f.replace("X_chunk", "").replace(".npy", "")) for f in chunk_files])
                                    for c in chunk_indices:
                                        key = (pair, tf, yr_name, c)
                                        self._shard_paths[key] = (tf_dir, c)
                                        tf_group_key = (pair, tf, yr_name)
                                        if tf_group_key not in self._tf_map:
                                            self._tf_map[tf_group_key] = []
                                        self._tf_map[tf_group_key].append(key)

                                        X, _, _, _ = load_shard(tf_dir, chunk_idx=c, mmap_mode="r")
                                        if X is not None:
                                            length = len(X)
                                            del X
                                            for row in range(length):
                                                self._index.append((p_idx, tf, key, row))

        else:
            # --- Legacy Fold-Based Manifold (folds/fold_N) ---
            for pair in self.pairs:
                p_idx = PAIR_INDEX.get(pair, 0)
                pair_root = None
                for root in self.shard_roots:
                    cand = self._resolve_pair_dir(root, pair)
                    if cand:
                        pair_root = root
                        break

                if pair_root is None:
                    continue

                for tf in self.active_timeframes:
                    if self.split == "val":
                        shard_dir = os.path.join(pair_root, pair, str(tf), "folds", "val")
                        key = (pair, tf, "val", -1)
                        X, y_dir, y_mag, _ts = load_shard(shard_dir, mmap_mode="r")
                        if X is not None:
                            self._shard_paths[key] = (shard_dir, -1)
                            length = len(X)
                            del X, y_dir, y_mag, _ts
                            for row in range(length):
                                self._index.append((p_idx, tf, key, row))
                    else:
                        max_fold = self.fold if self.fold is not None else 6
                        for f in range(1, max_fold + 1):
                            fold_name = f"fold_{f}"
                            shard_dir = os.path.join(pair_root, pair, str(tf), "folds", fold_name)
                            key = (pair, tf, fold_name, -1)
                            X, y_dir, y_mag, _ts = load_shard(shard_dir, mmap_mode="r")
                            if X is not None:
                                self._shard_paths[key] = (shard_dir, -1)
                                length = len(X)
                                del X, y_dir, y_mag, _ts
                                for row in range(length):
                                    self._index.append((p_idx, tf, key, row))

        self.all_samples = list(self._index)

        # Governor fractional sampling (train only, uniform chronological stride across all pairs/years)
        if self.is_train and self.sample_fraction < 1.0:
            stride = max(1, round(1.0 / max(1e-4, self.sample_fraction)))
            self._index = self.all_samples[::stride]

        if len(self._index) == 0:
            print(
                f" [ForexDataset] NOTICE: No matching shards found in {self.shard_roots} "
                f"for pairs={self.pairs}, TFs={self.active_timeframes} (Fold: {self.fold})."
            )

    def update_strategy(
        self,
        fraction: float | None = None,
        active_timeframes: list | None = None,
        size: float | int | tuple | None = None,
        stress: float | None = None,
        **kwargs
    ):
        """
        Governor hook: update sampling fraction or active timeframes mid-training.
        Mirrors MultiTaskDataset.update_strategy() interface.
        """
        rebuild = False
        if active_timeframes is not None and active_timeframes != self.active_timeframes:
            self.active_timeframes = active_timeframes
            rebuild = True
        if stress is not None:
            self.spread_stress_pips = stress

        if fraction is not None and fraction != self.sample_fraction:
            self.sample_fraction = fraction
            if not rebuild and hasattr(self, 'all_samples') and self.all_samples:
                stride = max(1, round(1.0 / max(1e-4, self.sample_fraction))) if (self.is_train and self.sample_fraction < 1.0) else 1
                self._index = self.all_samples[::stride]
                return

        if rebuild:
            self._build_index()

    def __len__(self) -> int:
        return max(1, len(self._index))

    def _get_shard_data(self, key: tuple):
        """Lazily retrieves or loads shard arrays for a specific key."""
        if getattr(self, '_shards', None) is None:
            self._shards = {}
        if key not in self._shards:
            tf_dir, chunk_idx = self._shard_paths[key]
            c_arg = chunk_idx if chunk_idx >= 0 else None
            self._shards[key] = load_shard(tf_dir, chunk_idx=c_arg, mmap_mode="r")
        return self._shards[key]

    def _get_alignment_map(self, primary_key: tuple, other_key: tuple):
        """Lazily computes or retrieves precalculated alignment row mapping between two shards."""
        if getattr(self, '_alignment_cache', None) is None:
            self._alignment_cache = {}
        pair_key = (primary_key, other_key)
        if pair_key in self._alignment_cache:
            return self._alignment_cache[pair_key]

        _, _, _, prim_ts = self._get_shard_data(primary_key)
        other_X, _, _, other_ts = self._get_shard_data(other_key)

        if other_X is None or len(other_X) == 0 or prim_ts is None or len(prim_ts) == 0:
            self._alignment_cache[pair_key] = None
            return None

        if (
            other_ts is not None and len(other_ts) > 1 and len(prim_ts) > 1 and
            prim_ts[-1] > prim_ts[0] and other_ts[-1] > other_ts[0]
        ):
            # Vectorized alignment precalculated once per shard pair (2ms vs 35M searches)
            aligned = np.searchsorted(other_ts, prim_ts, side='right') - 1
            aligned = np.clip(aligned, 0, len(other_X) - 1).astype(np.int64)
        else:
            prim_tf = primary_key[1]
            other_tf = other_key[1]
            ratio = prim_tf / other_tf
            rows = np.arange(len(prim_ts), dtype=np.float64)
            aligned = np.clip((rows * ratio).astype(np.int64), 0, len(other_X) - 1)

        self._alignment_cache[pair_key] = aligned
        return aligned

    def __getitem__(self, index: int):
        """
        Returns:
            tf_inputs: Dict[int -> Tensor[seq_len, features]] -- one entry per active TF
            labels:    Dict with 'direction' (long) and 'magnitude' (float32 [2])
            pair_idx:  Long tensor (scalar)
        """
        if len(self._index) == 0:
            # Dummy batch fallback to prevent dataloader crash on empty discovery
            tf_inputs = {tf: torch.zeros(TIMEFRAME_LOOKBACK.get(tf, 168), 14) for tf in self.active_timeframes}
            return (
                tf_inputs,
                {"direction": torch.tensor(1, dtype=torch.long),
                 "magnitude": torch.zeros(2, dtype=torch.float32)},
                torch.tensor(0, dtype=torch.long),
            )

        p_idx, tf, key, row = self._index[index]

        # ─── Branch A: Native High-Throughput Parquet Retrieval ─────────────
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "parquet":
            yr_name = key[1]
            meta = self._parquet_meta[yr_name]
            cache = self._parquet_caches[yr_name]
            pair_name = str(meta["pair"][row])
            current_ts = meta["timestamp"][row]

            target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
            raw_sample = cache.get_row_features(row)
            if raw_sample.shape[0] < target_len:
                pad = np.zeros((target_len - raw_sample.shape[0], raw_sample.shape[1]), dtype=raw_sample.dtype)
                raw_sample = np.concatenate([pad, raw_sample], axis=0)
            elif raw_sample.shape[0] > target_len:
                raw_sample = raw_sample[-target_len:]

            tf_inputs = {tf: torch.from_numpy(np.array(raw_sample, copy=True)).float()}

            # Cross-timeframe alignment via pre-extracted timestamps
            for other_tf in self.active_timeframes:
                if other_tf == tf:
                    continue
                other_target_len = TIMEFRAME_LOOKBACK.get(other_tf, 168)
                other_key = (pair_name, other_tf, yr_name)
                if other_key in self._parquet_tf_rows:
                    other_rows = self._parquet_tf_rows[other_key]
                    other_ts = self._parquet_tf_ts[other_key]
                    if len(other_rows) > 0 and len(other_ts) > 0:
                        aligned_idx = np.searchsorted(other_ts, current_ts, side="right") - 1
                        aligned_idx = max(0, min(aligned_idx, len(other_rows) - 1))
                        aligned_row = int(other_rows[aligned_idx])
                        other_sample = cache.get_row_features(aligned_row)
                        if other_sample.shape[0] < other_target_len:
                            pad = np.zeros((other_target_len - other_sample.shape[0], other_sample.shape[1]), dtype=other_sample.dtype)
                            other_sample = np.concatenate([pad, other_sample], axis=0)
                        elif other_sample.shape[0] > other_target_len:
                            other_sample = other_sample[-other_target_len:]
                        tf_inputs[other_tf] = torch.from_numpy(np.array(other_sample, copy=True)).float()
                    else:
                        tf_inputs[other_tf] = torch.zeros(other_target_len, raw_sample.shape[-1])
                else:
                    tf_inputs[other_tf] = torch.zeros(other_target_len, raw_sample.shape[-1])

            scale = PAIR_PIP_SCALE.get(pair_name, 1.0)
            mag_tp = float(meta["tp_pips"][row]) / scale
            mag_sl = float(meta["sl_pips"][row]) / scale
            if self.spread_stress_pips > 0.0:
                mag_tp = max(0.0, mag_tp - (self.spread_stress_pips / scale))
                mag_sl = mag_sl + (self.spread_stress_pips / scale)

            labels = {
                "direction": torch.tensor(int(meta["y_dir"][row]), dtype=torch.long),
                "magnitude": torch.tensor([mag_tp, mag_sl], dtype=torch.float32),
            }
            return (
                tf_inputs,
                labels,
                torch.tensor(p_idx, dtype=torch.long),
            )

        # ─── Branch B: Legacy .npy Shard Retrieval ──────────────────────────
        pair_name = str(key[0])
        yr_identifier = str(key[2]) if len(key) > 2 else ""

        X, y_dir, y_mag, _ = self._get_shard_data(key)

        # Primary timeframe tensor with shape invariant guarantee
        target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
        raw_sample = X[row]
        if raw_sample.shape[0] < target_len:
            pad = np.zeros((target_len - raw_sample.shape[0], raw_sample.shape[1]), dtype=raw_sample.dtype)
            raw_sample = np.concatenate([pad, raw_sample], axis=0)
        elif raw_sample.shape[0] > target_len:
            raw_sample = raw_sample[-target_len:]

        tf_inputs = {tf: torch.from_numpy(np.array(raw_sample, copy=True)).float()}

        # Cross-timeframe multi-scale alignment via O(1) precalculated map
        for other_tf in self.active_timeframes:
            if other_tf == tf:
                continue

            other_target_len = TIMEFRAME_LOOKBACK.get(other_tf, 168)
            other_group_key = (pair_name, other_tf, yr_identifier)
            candidate_keys = self._tf_map.get(other_group_key, [])

            if candidate_keys:
                other_key = candidate_keys[0]
                other_X, _, _, _ = self._get_shard_data(other_key)
                if other_X is None or len(other_X) == 0:
                    tf_inputs[other_tf] = torch.zeros(other_target_len, raw_sample.shape[-1])
                    continue

                alignment_map = self._get_alignment_map(key, other_key)
                if alignment_map is not None and row < len(alignment_map):
                    aligned_row = int(alignment_map[row])
                else:
                    ratio = tf / other_tf
                    aligned_row = min(int(row * ratio), len(other_X) - 1)

                other_sample = other_X[aligned_row]
                if other_sample.shape[0] < other_target_len:
                    pad = np.zeros((other_target_len - other_sample.shape[0], other_sample.shape[1]), dtype=other_sample.dtype)
                    other_sample = np.concatenate([pad, other_sample], axis=0)
                elif other_sample.shape[0] > other_target_len:
                    other_sample = other_sample[-other_target_len:]

                tf_inputs[other_tf] = torch.from_numpy(np.array(other_sample, copy=True)).float()
            else:
                tf_inputs[other_tf] = torch.zeros(other_target_len, raw_sample.shape[-1])

        # Magnitude scaling (Normalized Pip Units)
        scale = PAIR_PIP_SCALE.get(pair_name, 1.0)
        mag_tp = float(y_mag[row, 0]) / scale
        mag_sl = float(y_mag[row, 1]) / scale
        if self.spread_stress_pips > 0.0:
            mag_tp = max(0.0, mag_tp - (self.spread_stress_pips / scale))
            mag_sl = mag_sl + (self.spread_stress_pips / scale)

        labels = {
            "direction": torch.tensor(int(y_dir[row]), dtype=torch.long),
            "magnitude": torch.tensor([mag_tp, mag_sl], dtype=torch.float32),
        }

        return (
            tf_inputs,
            labels,
            torch.tensor(p_idx, dtype=torch.long),
        )
