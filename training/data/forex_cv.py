"""Purged and embargoed cross-validation splitting for financial time series.

Implements Purged Walk-Forward Cross-Validation and Purged K-Fold CV based on
Marcos López de Prado (2018), *Advances in Financial Machine Learning*, Chapter 7.

Eliminates two distinct mechanisms of data leakage in non-stationary financial series:
1. Label Overlap Leakage (Purging): Drops training observations whose future outcome
   horizon overlaps with the held-out validation or test interval.
2. Serial Correlation / Autoregressive Memory Leakage (Embargoing): Drops observations
   immediately subsequent to the validation fold to neutralize auto-correlated feature memory.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FoldSplit:
    """Index ranges and sample counts for one cross-validation fold."""

    fold_idx: int
    train_indices: Sequence[int]
    val_indices: Sequence[int]
    purged_count: int
    embargoed_count: int


class PurgedWalkForwardCV:
    """Walk-forward cross-validation splitter with label purging and post-validation embargo.

    Args:
        num_samples: Total number of sequential chronological samples.
        num_folds: Number of walk-forward validation folds (default: 6).
        warmup_fraction: Initial training window fraction before the first validation fold.
        purge_window: Forward lookahead bars in label barrier horizon (default: 10).
        embargo_pct: Series length fraction dropped after validation fold (default: 0.01).
    """

    def get_n_splits(self) -> int:
        """Return the number of validation folds."""
        return self.num_folds

    def __init__(
        self,
        num_samples: int,
        num_folds: int = 6,
        warmup_fraction: float = 0.3,
        purge_window: int = 10,
        embargo_pct: float = 0.01,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_folds <= 0:
            raise ValueError(f"num_folds must be positive, got {num_folds}")

        self.num_samples = int(num_samples)
        self.num_folds = int(num_folds)
        self.warmup_fraction = float(warmup_fraction)
        self.purge_window = max(0, purge_window)
        self.embargo_bars = max(0, math.ceil(num_samples * float(embargo_pct)))

    def split(self) -> Iterator[FoldSplit]:
        """Generate (train_indices, val_indices) splits across expanding walk-forward windows."""
        warmup_samples = max(10, int(self.num_samples * self.warmup_fraction))
        eval_samples = self.num_samples - warmup_samples

        if eval_samples < self.num_folds:
            raise ValueError(f"Insufficient samples ({eval_samples}) for {self.num_folds} folds.")

        fold_size = eval_samples // self.num_folds

        for fold in range(self.num_folds):
            val_start = warmup_samples + fold * fold_size
            val_end = val_start + fold_size if fold < self.num_folds - 1 else self.num_samples
            val_indices = list(range(val_start, val_end))

            # In walk-forward training, candidate training data precedes validation: [0, val_start)
            # Purging: remove training bars within purge_window of val_start
            train_end = max(0, val_start - self.purge_window)
            purged_count = val_start - train_end

            train_indices = list(range(train_end))
            embargoed_count = 0  # Preceding walk-forward does not evaluate after val_end

            yield FoldSplit(
                fold_idx=fold,
                train_indices=train_indices,
                val_indices=val_indices,
                purged_count=purged_count,
                embargoed_count=embargoed_count,
            )


class PurgedKFoldCV:
    """Purged and embargoed K-Fold cross validation for non-sequential financial models."""

    def get_n_splits(self) -> int:
        """Return the number of cross-validation splits."""
        return self.n_splits

    def __init__(
        self,
        num_samples: int,
        n_splits: int = 5,
        purge_window: int = 10,
        embargo_pct: float = 0.01,
    ) -> None:
        if num_samples <= 0 or n_splits <= 1:
            raise ValueError("num_samples and n_splits must be valid positive integers.")

        self.num_samples = int(num_samples)
        self.n_splits = n_splits
        self.purge_window = max(0, purge_window)
        self.embargo_bars = max(0, math.ceil(num_samples * float(embargo_pct)))

    def split(self) -> Iterator[FoldSplit]:
        """Generate purged and embargoed K-fold train/validation indices."""
        fold_size = self.num_samples // self.n_splits

        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else self.num_samples
            val_indices = set(range(val_start, val_end))

            # Determine purged indices (immediately preceding val_start)
            purge_start = max(0, val_start - self.purge_window)
            purged_indices = set(range(purge_start, val_start))

            # Determine embargoed indices (immediately succeeding val_end)
            embargo_end = min(self.num_samples, val_end + self.embargo_bars)
            embargoed_indices = set(range(val_end, embargo_end))

            excluded = val_indices | purged_indices | embargoed_indices
            train_indices = [i for i in range(self.num_samples) if i not in excluded]

            yield FoldSplit(
                fold_idx=fold,
                train_indices=train_indices,
                val_indices=sorted(val_indices),
                purged_count=len(purged_indices),
                embargoed_count=len(embargoed_indices),
            )
