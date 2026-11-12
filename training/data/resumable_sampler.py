"""Deterministic, epoch-seeded batch sampler that resumes mid-epoch without reading data.

The default ``shuffle=True`` loader draws an unseeded permutation, so a resumed epoch saw a
different sample order and the loop had to load, decode and discard every finished batch
before it reached the resume point. ``ResumableBatchSampler`` draws its permutation from
``seed + epoch`` and can start at batch ``K``: resuming slices the index list instead of
reading ``K`` batches, and the remaining batches are exactly the ones the interrupted run
would have trained on.

Under DDP every rank takes its strided share of the (padded) permutation, matching
``DistributedSampler``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

DEFAULT_SAMPLER_SEED = 42
"""Base seed of the epoch permutations (the Forex row-group sampler uses the same)."""


def distributed_rank_world() -> tuple[int, int]:
    """``(rank, world_size)`` of the initialised process group, else ``(0, 1)``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class ResumableBatchSampler(Sampler[list[int]]):
    """Batches of an epoch-seeded permutation, sharded per rank, optionally started at ``K``.

    ``set_epoch`` selects the permutation; ``set_start_batch`` makes the next iteration
    begin at that batch (once). ``len()`` is always the full epoch's batch count, so
    progress bars and resume steps keep counting from the epoch start.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = DEFAULT_SAMPLER_SEED,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.rank, self.world_size = distributed_rank_world()
        self.epoch = 0
        self._start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation of ``epoch``."""
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Begin the next iteration at ``start_batch`` (resets after that iteration)."""
        self._start_batch = max(0, int(start_batch))

    def _rank_indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        if self.world_size > 1 and indices:
            per_rank = math.ceil(len(indices) / self.world_size)
            total = per_rank * self.world_size
            indices = (indices * math.ceil(total / len(indices)))[:total]
            indices = indices[self.rank : total : self.world_size]
        return indices

    def _rank_len(self) -> int:
        if self.world_size > 1 and self.num_samples:
            return math.ceil(self.num_samples / self.world_size)
        return self.num_samples

    def __len__(self) -> int:
        samples = self._rank_len()
        if self.drop_last:
            return samples // self.batch_size
        return math.ceil(samples / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        indices = self._rank_indices()
        start, self._start_batch = self._start_batch, 0
        for batch_idx in range(start, len(self)):
            yield indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]


def loader_batch_size(loader: Any, fallback: int) -> int:
    """Batch size of ``loader``, also when it batches through a batch sampler."""
    size = getattr(loader, "batch_size", None) or getattr(
        getattr(loader, "batch_sampler", None), "batch_size", None
    )
    return int(size) if size else fallback


def set_loader_epoch(loader: Any, epoch: int) -> None:
    """Reseed the loader's sampler and batch sampler for ``epoch`` when they support it."""
    for source in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
        set_epoch = getattr(source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


def resume_start_batch(loader: Any, resume_step: int) -> int:
    """Start ``loader``'s next pass at ``resume_step`` without reading the skipped batches.

    Returns the batch index the pass starts at: ``resume_step`` when the loader batches
    through a :class:`ResumableBatchSampler`, else 0 (the caller skips batches itself).
    """
    batch_sampler = getattr(loader, "batch_sampler", None)
    if resume_step > 0 and isinstance(batch_sampler, ResumableBatchSampler):
        batch_sampler.set_start_batch(resume_step)
        return resume_step
    return 0
