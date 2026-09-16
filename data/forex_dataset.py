"""
LemGendary ForexDataset v3.0
==============================
PyTorch Dataset class supporting both year-based Walk-Forward manifolds
(ForexUniverse2019..2026) and legacy fold-sharded manifolds, with
chunked .npy loading, cross-timeframe alignment, and Governor integration.

Integrates with MultiTaskDataset pattern via task_type = "forex".

v3.0 Changes:
  - ParquetRowGroupCache replaced with a memmap-backed flat feature cache.
    On first construction, every row group is decoded once and written to a
    single .flat file plus per-row offset/shape tables. Subsequent opens
    mmap those files, so get_row_features(i) is a pure numpy slice — no
    pyarrow decode, no decompression, no LRU eviction, no per-row copies.
    First-run build cost is a few minutes; every subsequent run opens in
    milliseconds.
  - __getitem__ now uses torch.from_numpy on the memmap slice instead of
    torch.tensor(), eliminating a per-row float32 copy (×6 TFs per sample).

v2.3 Changes (retained in spirit):
  - Decoded row groups are per-row numpy arrays, detached from pyarrow
    chunked-array machinery. v3.0 pushes this one step further by persisting
    the decoded layout to disk.

v2.2 Changes (retained):
  - Exposes _parquet_meta[year]['rg_of_row'] so RowGroupAwareSampler can
    shuffle at the row-group level (raises LRU hit rate from ~2% to ~99%).

v2.1 Changes (retained):
  - O(1) aligned-row lookup via precomputed alignment maps.
  - Eliminated double-copy on the legacy .npy path.
"""

import os
import json
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
    Memmap-backed feature cache for a single year's ForexUniverse parquet.

    On first construction, decodes every row group once and writes:
        <stem>.v<N>.features.flat    -- concatenated float32 payload
        <stem>.v<N>.offsets.npy      -- int64 start offsets, length N+1
        <stem>.v<N>.shapes.npy       -- int32 (seq_len, n_features) per row
        <stem>.v<N>.meta.json        -- source parquet size + mtime for validity

    Subsequent opens mmap those files. get_row_features(i) becomes a
    ~100ns numpy slice. Cache is invalidated automatically when the source
    parquet's size or mtime changes, or when CACHE_VERSION is bumped.

    Disk cost: approximately 1.2–1.5× the source parquet file size.
    RAM cost: page cache only; the OS reclaims cold pages under pressure.
    """
    CACHE_VERSION = 1

    def __init__(self, parquet_path: str, cache_dir: str | None = None):
        import pyarrow.parquet as pq
        self.parquet_path = parquet_path
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(parquet_path), "_forex_cache")
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(parquet_path))[0]
        base = os.path.join(cache_dir, f"{stem}.v{self.CACHE_VERSION}")
        self._base = base
        self._flat_path    = base + ".features.flat"
        self._offsets_path = base + ".offsets.npy"
        self._shapes_path  = base + ".shapes.npy"
        self._meta_path    = base + ".meta.json"

        if not self._cache_valid():
            self._build_cache(parquet_path)
            self._write_meta()

        self.row_offsets = np.load(self._offsets_path, mmap_mode="r")
        self.row_shapes  = np.load(self._shapes_path,  mmap_mode="r")
        total_floats = int(self.row_offsets[-1])
        self.flat = np.memmap(self._flat_path, dtype=np.float32, mode="r", shape=(total_floats,))

        self.pf = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.pf.num_row_groups
        self.metadata = self.pf.metadata
        self.rg_starts = []
        curr = 0
        for i in range(self.num_row_groups):
            self.rg_starts.append(curr)
            curr += self.metadata.row_group(i).num_rows
        self.total_rows = curr
        assert self.total_rows == len(self.row_shapes), (
            f"[CACHE] row count mismatch: parquet={self.total_rows} cache={len(self.row_shapes)}"
        )

    def _cache_valid(self) -> bool:
        if not (os.path.exists(self._flat_path)
                and os.path.exists(self._offsets_path)
                and os.path.exists(self._shapes_path)
                and os.path.exists(self._meta_path)):
            return False
        try:
            st = os.stat(self.parquet_path)
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("version") != self.CACHE_VERSION:
                return False
            if meta.get("src_size") != st.st_size:
                return False
            if int(meta.get("src_mtime", 0)) != int(st.st_mtime):
                return False
            return True
        except Exception:
            return False

    def _write_meta(self):
        st = os.stat(self.parquet_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": self.CACHE_VERSION,
                "src_size": st.st_size,
                "src_mtime": int(st.st_mtime),
                "src_path": os.path.abspath(self.parquet_path),
            }, f)

    def _build_cache(self, parquet_path: str):
        import pyarrow.parquet as pq
        print(f" [CACHE] Building memmap cache for {os.path.basename(parquet_path)} (one-time)...")
        pf = pq.ParquetFile(parquet_path)
        num_rgs = pf.num_row_groups

        tmp_flat    = self._flat_path + ".tmp"
        tmp_offsets = self._offsets_path + ".tmp.npy"
        tmp_shapes  = self._shapes_path  + ".tmp.npy"

        cursor = 0
        offsets = [0]
        shapes = []

        with open(tmp_flat, "wb") as fout:
            for rg_idx in range(num_rgs):
                rg_tbl = pf.read_row_group(rg_idx, columns=["features", "seq_len", "n_features"])
                feats    = rg_tbl["features"].combine_chunks()
                seq_lens = rg_tbl["seq_len"].to_numpy()
                n_feats  = rg_tbl["n_features"].to_numpy()
                for i in range(len(seq_lens)):
                    arr = np.frombuffer(feats[i].as_buffer(), dtype=np.float32)
                    fout.write(arr.tobytes())
                    cursor += arr.size
                    offsets.append(cursor)
                    shapes.append((int(seq_lens[i]), int(n_feats[i])))

        os.replace(tmp_flat, self._flat_path)
        np.save(tmp_offsets, np.asarray(offsets, dtype=np.int64))
        os.replace(tmp_offsets, self._offsets_path)
        np.save(tmp_shapes, np.asarray(shapes, dtype=np.int32))
        os.replace(tmp_shapes, self._shapes_path)

        print(f" [CACHE] Done: {len(shapes)} rows, {cursor} floats ({cursor * 4 / 1e9:.2f} GB).")

    def get_row_features(self, global_row_idx: int) -> np.ndarray:
        o1 = int(self.row_offsets[global_row_idx])
        o2 = int(self.row_offsets[global_row_idx + 1])
        s  = self.row_shapes[global_row_idx]
        return self.flat[o1:o2].reshape(int(s[0]), int(s[1]))


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
            colab_base = "/content/LemGendaryDatasets"
            if os.path.exists(colab_base):
                for d in os.listdir(colab_base):
                    if "forex" in d.lower():
                        cand_forex = os.path.join(colab_base, d, "forex")
                        cand = cand_forex if os.path.isdir(cand_forex) else os.path.join(colab_base, d)
                        if os.path.isdir(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)
            base_drive = "/content/drive/MyDrive/LemGendaryDatasets"
            if os.path.exists(base_drive):
                for d in os.listdir(base_drive):
                    if "forex" in d.lower():
                        cand_forex = os.path.join(base_drive, d, "forex")
                        cand = cand_forex if os.path.isdir(cand_forex) else os.path.join(base_drive, d)
                        if os.path.isdir(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)
            if not resolved_roots:
                print(f"\n[ERROR] Colab requires dataset packages at {colab_base} or {base_drive}")
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
        self._parquet_alignment_cache = {}
        self._alignment_cache = {}
        self._shards = {}

        self._build_index()

    def _resolve_pair_dir(self, parent_dir: str, pair: str) -> str | None:
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

    def _build_index(self):
        """Build flat sample index over all pairs x active_timeframes x rows."""
        self._shard_paths = {}
        self._shards = {}
        self._index = []
        self._tf_map = {}

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

                    rg_of_row = np.empty(cache.total_rows, dtype=np.int32)
                    for rg_idx in range(cache.num_row_groups):
                        rg_start = cache.rg_starts[rg_idx]
                        rg_end = (cache.rg_starts[rg_idx + 1]
                                  if rg_idx + 1 < cache.num_row_groups
                                  else cache.total_rows)
                        rg_of_row[rg_start:rg_end] = rg_idx

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
                        "rg_of_row": rg_of_row,
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
        if key not in self._shards:
            tf_dir, chunk_idx = self._shard_paths[key]
            c_arg = chunk_idx if chunk_idx >= 0 else None
            self._shards[key] = load_shard(tf_dir, chunk_idx=c_arg, mmap_mode="r")
        return self._shards[key]

    def _get_alignment_map(self, primary_key: tuple, other_key: tuple):
        """Lazily computes or retrieves precalculated alignment row mapping between two shards."""
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

    def _get_parquet_alignment(self, pair_name: str, primary_tf: int, other_tf: int, yr_name: str):
        align_key = (pair_name, primary_tf, other_tf, yr_name)
        if align_key in self._parquet_alignment_cache:
            return self._parquet_alignment_cache[align_key]

        other_key = (pair_name, other_tf, yr_name)
        primary_key = (pair_name, primary_tf, yr_name)

        other_rows = self._parquet_tf_rows.get(other_key)
        other_ts = self._parquet_tf_ts.get(other_key)
        prim_ts = self._parquet_tf_ts.get(primary_key)

        if (other_rows is None or other_ts is None or prim_ts is None
                or len(other_rows) == 0 or len(other_ts) == 0 or len(prim_ts) == 0):
            self._parquet_alignment_cache[align_key] = None
            return None

        mapped = np.searchsorted(other_ts, prim_ts, side="right") - 1
        mapped = np.clip(mapped, 0, len(other_rows) - 1).astype(np.int64)
        result = (other_rows, mapped)
        self._parquet_alignment_cache[align_key] = result
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

    def __getitem__(self, index: int):
        """
        Returns:
            tf_inputs: Dict[int -> Tensor[seq_len, features]]
            labels:    Dict with 'direction' (long) and 'magnitude' (float32 [2])
            pair_idx:  Long tensor (scalar)
        """
        if len(self._index) == 0:
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

            target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
            raw_sample = self._fit_window(cache.get_row_features(row), target_len)

            # 2026 v3.0: memmap slice aliased into a torch tensor (zero-copy).
            # PyTorch's default collate stacks into fresh storage before the
            # batch crosses the worker boundary, so the memmap view is
            # short-lived. Falls back to a defensive copy if the slice is not
            # already contiguous float32.
            if raw_sample.dtype == np.float32 and raw_sample.flags["C_CONTIGUOUS"]:
                primary_tensor = torch.from_numpy(raw_sample)
            else:
                primary_tensor = torch.from_numpy(np.ascontiguousarray(raw_sample, dtype=np.float32))
            tf_inputs = {tf: primary_tensor}

            n_features = raw_sample.shape[1]

            for other_tf in self.active_timeframes:
                if other_tf == tf:
                    continue
                other_target_len = TIMEFRAME_LOOKBACK.get(other_tf, 168)
                align = self._get_parquet_alignment(pair_name, tf, other_tf, yr_name)
                if align is None:
                    tf_inputs[other_tf] = torch.zeros(other_target_len, n_features)
                    continue

                other_rows, mapped = align
                prim_rows = self._parquet_tf_rows.get((pair_name, tf, yr_name))
                if prim_rows is None or len(prim_rows) == 0:
                    tf_inputs[other_tf] = torch.zeros(other_target_len, n_features)
                    continue

                import bisect as _bisect
                pos = _bisect.bisect_left(prim_rows, row)
                if pos >= len(prim_rows) or int(prim_rows[pos]) != int(row):
                    tf_inputs[other_tf] = torch.zeros(other_target_len, n_features)
                    continue

                aligned_row = int(other_rows[int(mapped[pos])])
                other_sample = self._fit_window(cache.get_row_features(aligned_row), other_target_len)

                if other_sample.dtype == np.float32 and other_sample.flags["C_CONTIGUOUS"]:
                    tf_inputs[other_tf] = torch.from_numpy(other_sample)
                else:
                    tf_inputs[other_tf] = torch.from_numpy(np.ascontiguousarray(other_sample, dtype=np.float32))

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

        target_len = TIMEFRAME_LOOKBACK.get(tf, 168)
        raw_sample = self._fit_window(np.asarray(X[row]), target_len)

        if raw_sample.dtype == np.float32 and raw_sample.flags["C_CONTIGUOUS"]:
            tf_inputs = {tf: torch.from_numpy(raw_sample)}
        else:
            tf_inputs = {tf: torch.from_numpy(np.ascontiguousarray(raw_sample, dtype=np.float32))}

        n_features = raw_sample.shape[1]

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
                    tf_inputs[other_tf] = torch.zeros(other_target_len, n_features)
                    continue

                alignment_map = self._get_alignment_map(key, other_key)
                if alignment_map is not None and row < len(alignment_map):
                    aligned_row = int(alignment_map[row])
                else:
                    ratio = tf / other_tf
                    aligned_row = min(int(row * ratio), len(other_X) - 1)

                other_sample = self._fit_window(np.asarray(other_X[aligned_row]), other_target_len)
                if other_sample.dtype == np.float32 and other_sample.flags["C_CONTIGUOUS"]:
                    tf_inputs[other_tf] = torch.from_numpy(other_sample)
                else:
                    tf_inputs[other_tf] = torch.from_numpy(np.ascontiguousarray(other_sample, dtype=np.float32))
            else:
                tf_inputs[other_tf] = torch.zeros(other_target_len, n_features)

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