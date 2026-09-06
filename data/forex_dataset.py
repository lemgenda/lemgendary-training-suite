"""
LemGendary ForexDataset v1.0
==============================
PyTorch Dataset class that loads pre-built .npy shards from mt5_pipeline.py
and feeds them to the training loop with Governor-aligned fractional sampling.

Integrates with MultiTaskDataset pattern via task_type = "forex".
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Literal

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
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD",
    "USDCAD", "USDCHF", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP",
    "XAGUSD", "USOIL",
    "US500", "USTEC", "GER40"
]
PAIR_INDEX = {p: i for i, p in enumerate(EXTENDED_PAIRS)}

PAIR_PIP_SCALE = {
    "EURUSD": 1.0, "GBPUSD": 1.0, "USDJPY": 1.0, "USDCAD": 1.0,
    "USDCHF": 1.0, "AUDUSD": 1.0, "NZDUSD": 1.0,
    "EURGBP": 1.0, "EURJPY": 1.0, "GBPJPY": 1.0,
    "XAGUSD": 5.0, "USOIL": 5.0, "XAUUSD": 10.0,
    "US500": 20.0, "GER40": 20.0, "USTEC": 40.0
}

def load_shard(shard_dir: str, mmap_mode: Literal['c', 'r', 'r+', 'w+'] | None=None) -> tuple:
    X_path = os.path.join(shard_dir, 'X.npy')
    ydir_path = os.path.join(shard_dir, 'y_dir.npy')
    ymag_path = os.path.join(shard_dir, 'y_mag.npy')
    ts_path = os.path.join(shard_dir, 'timestamps.npy')
    if not (os.path.exists(X_path) and os.path.exists(ydir_path) and os.path.exists(ymag_path)):
        return (None, None, None, None)
    X = np.load(X_path, mmap_mode=mmap_mode)
    y_dir = np.load(ydir_path, mmap_mode=mmap_mode)
    y_mag = np.load(ymag_path, mmap_mode=mmap_mode)
    timestamps = np.load(ts_path, mmap_mode=mmap_mode) if os.path.exists(ts_path) else None
    return (X, y_dir, y_mag, timestamps)

# [LemGendary Forex Suite v1.0 - SYNC_ID: FOREX_02]


class ForexDataset(Dataset):
    """
    Multi-pair, multi-timeframe Forex Dataset for the LemGendary Training Suite.

    Loads windowed OHLCV + indicator samples from pre-built .npy shards.
    Supports Governor-aligned fractional sampling (sample_fraction) and
    curriculum timeframe expansion (active_timeframes).

    Args:
        shard_root:         Root directory containing .npy shards (data/forex/).
        pairs:              List of currency pair symbols to include.
        active_timeframes:  List of active timeframe rungs (minutes).
        is_train:           True for training split, False for validation.
        sample_fraction:    Fraction of training samples to use (Governor managed).
    """

    # task_type must be "forex" for train.py branching
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
# --- Multi-Environment Resolution (Multi-Root Distributed Loader) ---
        import yaml
        unified_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unified_models_v2.yaml")
        gdrive_ids = []
        if os.path.exists(unified_path):
            with open(unified_path, 'r') as f:
                um = yaml.safe_load(f)
                gdrive_ids = um.get("forex_predictor", {}).get("google_drive_dataset_ids", [])

        env = os.environ.get("ENV", "local")
        import sys
        if 'colab' in sys.modules or os.path.exists('/content'):
            env = 'colab'
        elif os.path.exists('/kaggle'):
            env = 'kaggle'

        resolved_roots = []
        if shard_root and os.path.exists(shard_root):
            resolved_roots.append(shard_root)
        
        if env == 'colab':
            base_drive = "/content/drive/MyDrive/LemGendaryDatasets"
            if os.path.exists(base_drive):
                for d in os.listdir(base_drive):
                    if "Forex" in d:
                        cand = os.path.join(base_drive, d, "forex")
                        if os.path.exists(cand):
                            resolved_roots.append(cand)
            if not resolved_roots:
                print(f"\n[ERROR] Colab requires Google Drive dataset modular packages at {base_drive}")
                import sys; sys.exit(1)
                
        elif env == 'kaggle':
            if os.path.exists('/kaggle/input'):
                for root, dirs, files in os.walk('/kaggle/input'):
                    if 'forex' in dirs:
                        resolved_roots.append(os.path.join(root, 'forex'))
                if not resolved_roots:
                    print(f"\n[WARNING] No attached forex datasets found in Kaggle /kaggle/input!")
            
        else: # Local
            base_search = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            ]
            for bs in base_search:
                if os.path.exists(bs):
                    for d in os.listdir(bs):
                        if "Forex" in d or d == "forex":
                            cand = os.path.join(bs, d, "forex") if "Forex" in d else os.path.join(bs, d)
                            if os.path.exists(cand) and cand not in resolved_roots:
                                resolved_roots.append(cand)
            
            if not resolved_roots and gdrive_ids:
                print(f"\n[DATA] Local forex dataset not found. Attempting gdown from {len(gdrive_ids)} Google Drive IDs...")
                import subprocess, zipfile
                base_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets")
                os.makedirs(base_out, exist_ok=True)
                for gid in gdrive_ids:
                    dest = os.path.join(base_out, f"dataset_{gid}.zip")
                    subprocess.run([sys.executable, "-m", "gdown", "--id", gid, "-O", dest])
                    if os.path.exists(dest):
                        with zipfile.ZipFile(dest, 'r') as zip_ref:
                            zip_ref.extractall(base_out)
                        os.remove(dest)
                for d in os.listdir(base_out):
                    if "Forex" in d:
                        cand = os.path.join(base_out, d, "forex")
                        if os.path.exists(cand) and cand not in resolved_roots:
                            resolved_roots.append(cand)

        self.shard_roots = resolved_roots
        if pairs:
            self.pairs = pairs
        elif self.shard_roots:
            disk_pairs = []
            for root in self.shard_roots:
                disk_pairs.extend([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.') and d in PAIR_INDEX])
            self.pairs = list(set(disk_pairs)) if disk_pairs else EXTENDED_PAIRS
        else:
            self.pairs = EXTENDED_PAIRS
        self.active_timeframes  = active_timeframes or [1, 5, 15, 60, 240, 1440]
        self.is_train           = is_train
        self.sample_fraction    = sample_fraction
        self.split              = "train" if is_train else "val"
        self.fold               = fold
        self.spread_stress_pips = spread_stress_pips

        # Auto-detect available timeframes across all active pairs
        available_tfs = None
        for pair in self.pairs:
            pair_root = None
            for root in self.shard_roots:
                if os.path.exists(os.path.join(root, pair)):
                    pair_root = root
                    break
            if pair_root:
                tfs_for_pair = set()
                pair_dir = os.path.join(pair_root, pair)
                for tf_dir in os.listdir(pair_dir):
                    if os.path.isdir(os.path.join(pair_dir, tf_dir)) and tf_dir.isdigit():
                        tfs_for_pair.add(int(tf_dir))
                if available_tfs is None:
                    available_tfs = tfs_for_pair
                else:
                    available_tfs = available_tfs.intersection(tfs_for_pair)
        
        if available_tfs is not None:
            self.active_timeframes = [tf for tf in self.active_timeframes if tf in available_tfs]

        # Build flat sample index: (pair_idx, tf, shard_row_idx)
        self._build_index()

    # ─────────────────────────────────────────────────────────────────────────
    # Index building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Build flat sample index over all pairs × active_timeframes × rows."""
        self._shard_paths = {} # (pair, tf) -> shard_dir
        self._shards = {}      # Populated lazily in __getitem__ by each worker
        self._index  = []     # [(pair_idx, tf, key, row_idx), ...]

        for pair in self.pairs:
            p_idx = PAIR_INDEX.get(pair, 0)
            pair_root = None
            for root in self.shard_roots:
                if os.path.exists(os.path.join(root, pair)):
                    pair_root = root
                    break
            
            if pair_root is None:
                print(f"\n[ERROR] Pair {pair} requested but not found in any attached dataset roots!")
                print(f"Attached roots: {self.shard_roots}")
                import sys; sys.exit(1)

            for tf in self.active_timeframes:
                if self.split == "val":
                    # Load only global validation
                    shard_dir = os.path.join(pair_root, pair, str(tf), "folds", "val")
                    key = (pair, tf, "val")
                    X, y_dir, y_mag, _timestamps = load_shard(shard_dir, mmap_mode="r")
                    if X is not None:
                        self._shard_paths[key] = shard_dir
                        length = len(X)
                        del X, y_dir, y_mag, _timestamps
                        for row in range(length):
                            self._index.append((p_idx, tf, key, row))
                else:
                    # Train split: concatenate all chronological folds up to self.fold
                    max_fold = self.fold if self.fold is not None else 6
                    for f in range(1, max_fold + 1):
                        fold_name = f"fold_{f}"
                        shard_dir = os.path.join(pair_root, pair, str(tf), "folds", fold_name)
                        key = (pair, tf, fold_name)
                        X, y_dir, y_mag, _timestamps = load_shard(shard_dir, mmap_mode="r")
                        if X is not None:
                            self._shard_paths[key] = shard_dir
                            length = len(X)
                            del X, y_dir, y_mag, _timestamps
                            for row in range(length):
                                self._index.append((p_idx, tf, key, row))

        self.all_samples = list(self._index)
        # Governor fractional sampling (train only, chronological order preserved)
        if self.is_train and self.sample_fraction < 1.0:
            n = max(1, int(len(self._index) * self.sample_fraction))
            self._index = self._index[:n]

        if len(self._index) == 0:
            print(
                f" [ForexDataset] NOTICE: No matching shards found in {self.shard_roots} "
                f"for pairs={self.pairs}, TFs={self.active_timeframes}."
            )

    def update_strategy(self, fraction: float | None = None, active_timeframes: list | None = None, size: float | int | tuple | None = None, **kwargs):
        """
        Governor hook: update sampling fraction or active timeframes mid-training.
        Mirrors MultiTaskDataset.update_strategy() interface.
        """
        if active_timeframes is not None:
            self.active_timeframes = active_timeframes
            self._build_index()
        elif fraction is not None:
            self.sample_fraction = fraction
            self._build_index()

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset Protocol
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return max(1, len(self._index))

    def __getitem__(self, index: int):
        """
        Returns:
            tf_inputs:  Dict[int → Tensor[seq_len, features]]  — one entry per active TF
            labels:     Dict with 'direction' (long) and 'magnitude' (float32 [2])
            pair_idx:   Long tensor (scalar)
        """
        if len(self._index) == 0:
            # No data loaded yet — return dummy batch so training loop doesn't crash
            tf_inputs = {tf: torch.zeros(TIMEFRAME_LOOKBACK[tf], 10) for tf in self.active_timeframes}
            return (
                tf_inputs,
                {"direction": torch.tensor(1, dtype=torch.long),
                 "magnitude": torch.zeros(2, dtype=torch.float32)},
                torch.tensor(0, dtype=torch.long),
            )

        p_idx, tf, key, row = self._index[index]
        if getattr(self, '_shards', None) is None:
            self._shards = {}
            
        if key not in self._shards:
            self._shards[key] = load_shard(self._shard_paths[key], mmap_mode="r")
        X, y_dir, y_mag, timestamps = self._shards[key]

        # Primary timeframe features
        tf_inputs = {tf: torch.from_numpy(np.array(X[row])).float()}

        # For multi-timeframe mode: load aligned bars from other TFs
        # Priority: timestamp-based alignment (precise), fallback to ratio-based (legacy shards)
        for other_tf in self.active_timeframes:
            if other_tf == tf:
                continue
            pair_name = self.pairs[p_idx] if p_idx < len(self.pairs) else "EURUSD"
            other_key = (pair_name, other_tf)
            if other_key in self._shard_paths:
                if other_key not in self._shards:
                    self._shards[other_key] = load_shard(self._shard_paths[other_key], mmap_mode="r")
                other_X, _, _, other_ts = self._shards[other_key]
                if other_X is None:
                    tf_inputs[other_tf] = torch.zeros(TIMEFRAME_LOOKBACK[other_tf], X.shape[-1])
                    continue
                if timestamps is not None and other_ts is not None:
                    # Precise alignment: find the latest other-TF bar that is <= current bar's timestamp
                    current_ts = int(timestamps[row])
                    aligned_row = int(np.searchsorted(other_ts, current_ts, side='right')) - 1
                    aligned_row = max(0, min(aligned_row, len(other_X) - 1))
                else:
                    # Legacy fallback: proportional ratio alignment
                    ratio = tf / other_tf
                    aligned_row = min(int(row * ratio), len(other_X) - 1)
                tf_inputs[other_tf] = torch.from_numpy(np.array(other_X[aligned_row])).float()
            else:
                # Pad with zeros if this TF shard not loaded yet
                tf_inputs[other_tf] = torch.zeros(TIMEFRAME_LOOKBACK[other_tf], X.shape[-1])

        pair_name = self.pairs[p_idx] if p_idx < len(self.pairs) else "EURUSD"
        scale = PAIR_PIP_SCALE.get(pair_name, 1.0)
        mag_tp = float(y_mag[row, 0]) / scale
        mag_sl = float(y_mag[row, 1]) / scale
        if self.spread_stress_pips > 0.0:
            # Spread friction reduces net TP gain and increases effective SL risk
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
