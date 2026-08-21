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

from data.mt5_pipeline import (
    TIMEFRAME_LOOKBACK,
    TIMEFRAME_RUNGS,
    MAJOR_PAIRS,
    EXTENDED_PAIRS,
    PAIR_INDEX,
    load_shard,
)

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
        # Auto-resolve shard_root across compiled manifold and local paths
        candidates = [
            shard_root,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets", "LemGendizedForexPredictorLarge", "forex"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "LemGendaryDatasets", "LemGendizedForexPredictorLarge"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "forex"),
        ]
        resolved_root = shard_root
        for cand in candidates:
            if cand and os.path.exists(cand) and any(os.path.isdir(os.path.join(cand, d)) for d in os.listdir(cand) if not d.startswith('.')):
                resolved_root = os.path.normpath(cand)
                break

        self.shard_root = resolved_root
        if pairs:
            self.pairs = pairs
        elif os.path.exists(self.shard_root):
            disk_pairs = [d for d in os.listdir(self.shard_root) if os.path.isdir(os.path.join(self.shard_root, d)) and not d.startswith('.') and d in PAIR_INDEX]
            self.pairs = disk_pairs if disk_pairs else EXTENDED_PAIRS
        else:
            self.pairs = EXTENDED_PAIRS
        self.active_timeframes  = active_timeframes or [1, 5, 15, 60, 240, 1440]
        self.is_train           = is_train
        self.sample_fraction    = sample_fraction
        self.split              = "train" if is_train else "val"
        self.fold               = fold
        self.spread_stress_pips = spread_stress_pips

        # Build flat sample index: (pair_idx, tf, shard_row_idx)
        self._build_index()

    # ─────────────────────────────────────────────────────────────────────────
    # Index building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_index(self):
        """Build flat sample index over all pairs × active_timeframes × rows."""
        self._shards = {}     # (pair, tf) → (X, y_dir, y_mag)
        self._index  = []     # [(pair_idx, tf, key, row_idx), ...]

        for pair in self.pairs:
            p_idx = PAIR_INDEX.get(pair, 0)
            for tf in self.active_timeframes:
                # Check for fold-specific shard first if fold is specified
                if self.fold is not None:
                    shard_dir = os.path.join(self.shard_root, pair, str(tf), "folds", f"fold_{self.fold}", self.split)
                    if not os.path.exists(shard_dir):
                        shard_dir = os.path.join(self.shard_root, pair, str(tf), self.split)
                else:
                    shard_dir = os.path.join(self.shard_root, pair, str(tf), self.split)

                X, y_dir, y_mag = load_shard(shard_dir, mmap_mode="r")
                if X is None:
                    # Shard not yet downloaded — skip silently during build
                    continue
                key = (pair, tf)
                self._shards[key] = (X, y_dir, y_mag)
                for row in range(len(X)):
                    self._index.append((p_idx, tf, key, row))

        self.all_samples = list(self._index)
        # Governor fractional sampling (train only, chronological order preserved)
        if self.is_train and self.sample_fraction < 1.0:
            n = max(1, int(len(self._index) * self.sample_fraction))
            self._index = self._index[:n]

        if len(self._index) == 0:
            print(
                f" [ForexDataset] NOTICE: No matching shards found in {self.shard_root} "
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
        X, y_dir, y_mag = self._shards[key]

        # Primary timeframe features
        tf_inputs = {tf: torch.from_numpy(np.array(X[row])).float()}

        # For multi-timeframe mode: attempt to load aligned bars from other TFs
        # (During M1-only FOUNDATION phase, only tf is populated)
        for other_tf in self.active_timeframes:
            if other_tf == tf:
                continue
            other_key = (self.pairs[p_idx] if p_idx < len(self.pairs) else "EURUSD", other_tf)
            if other_key in self._shards:
                other_X = self._shards[other_key][0]
                # Best-effort alignment: use same relative position (not timestamp-aligned)
                # TODO: Add proper UTC timestamp alignment once MT5 data is available
                aligned_row = min(row, len(other_X) - 1)
                tf_inputs[other_tf] = torch.from_numpy(np.array(other_X[aligned_row])).float()
            else:
                # Pad with zeros if this TF shard not loaded yet
                tf_inputs[other_tf] = torch.zeros(TIMEFRAME_LOOKBACK[other_tf], X.shape[-1])

        mag_tp = float(y_mag[row, 0])
        mag_sl = float(y_mag[row, 1])
        if self.spread_stress_pips > 0.0:
            # Spread friction reduces net TP gain and increases effective SL risk
            mag_tp = max(0.0, mag_tp - self.spread_stress_pips)
            mag_sl = mag_sl + self.spread_stress_pips

        labels = {
            "direction": torch.tensor(int(y_dir[row]), dtype=torch.long),
            "magnitude": torch.tensor([mag_tp, mag_sl], dtype=torch.float32),
        }

        return (
            tf_inputs,
            labels,
            torch.tensor(p_idx, dtype=torch.long),
        )
