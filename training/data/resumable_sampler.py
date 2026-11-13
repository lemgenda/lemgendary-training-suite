"""Deterministic, epoch-seeded batch sampler that resumes mid-epoch without reading data.

The default ``shuffle=True`` loader draws an unseeded permutation, so a resumed epoch saw a
different sample order and the loop had to load, decode and discard every finished batch
before it reached the resume point. ``ResumableBatchSampler`` draws its permutation from
``seed + epoch`` and can start at batch ``K``: resuming slices the index list instead of
reading ``K`` batches, and the remaining batches are exactly the ones the interrupted run
would have trained on.

Under DDP every rank takes its strided share of the (padded) permutation, matching
``DistributedSampler``.

After :meth:`ResumableBatchSampler.set_shard_window` the epoch permutation is the
shard-window order of :func:`training.data.locality.shard_window_order` instead of a
uniform one, so the ranks read the same few shards at a time; it is drawn from
``seed + epoch`` too.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from training.data.locality import ShardWindow

DEFAULT_SAMPLER_SEED = 42
"""Base seed of the epoch permutations (the Forex row-group sampler uses the same)."""


def distributed_rank_world() -> tuple[int, int]:
    """``(rank, world_size)`` of the initialised process group, else ``(0, 1)``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@dataclass
class EpochOrder:
    """How an epoch's sample order is drawn: seeded shuffle, shard windows or identity."""

    shuffle: bool = True
    seed: int = DEFAULT_SAMPLER_SEED
    epoch: int = 0
    shard_window: ShardWindow | None = None

    def indices(self, num_samples: int) -> list[int]:
        """The sample order of the current epoch over ``num_samples`` samples."""
        if self.shuffle and self.shard_window is not None:
            return self.shard_window.order(self.seed + self.epoch)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            return torch.randperm(num_samples, generator=generator).tolist()
        return list(range(num_samples))


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
        self.drop_last = drop_last
        self._order = EpochOrder(shuffle=shuffle, seed=int(seed))
        self._topology = distributed_rank_world()
        self._start_batch = 0

    @property
    def shuffle(self) -> bool:
        """Whether epochs are permuted (else the identity order)."""
        return self._order.shuffle

    @property
    def seed(self) -> int:
        """Base seed; epoch ``e`` draws its permutation from ``seed + e``."""
        return self._order.seed

    @property
    def epoch(self) -> int:
        """The epoch whose permutation the next iteration yields."""
        return self._order.epoch

    @property
    def shard_window(self) -> ShardWindow | None:
        """The shard-window ordering of shuffled epochs, if any."""
        return self._order.shard_window

    @property
    def rank(self) -> int:
        """This process's DDP rank (0 without a process group)."""
        return self._topology[0]

    @rank.setter
    def rank(self, rank: int) -> None:
        self._topology = (int(rank), self._topology[1])

    @property
    def world_size(self) -> int:
        """DDP world size (1 without a process group)."""
        return self._topology[1]

    @world_size.setter
    def world_size(self, world_size: int) -> None:
        self._topology = (self._topology[0], int(world_size))

    def set_shard_window(self, window: ShardWindow | None) -> None:
        """Order shuffled epochs by ``window`` (``None`` restores the uniform permutation)."""
        if window is not None and len(window.keys) != self.num_samples:
            raise ValueError(f"{len(window.keys)} shard keys for {self.num_samples} samples")
        self._order.shard_window = window

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation of ``epoch``."""
        self._order.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        """Begin the next iteration at ``start_batch`` (resets after that iteration)."""
        self._start_batch = max(0, int(start_batch))

    def _rank_indices(self) -> list[int]:
        indices = self._order.indices(self.num_samples)
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
    """Reseed ``loader`` for ``epoch``: sampler, batch sampler, dataset and generator.

    The sampler and batch sampler draw the epoch permutation, the dataset derives the
    per-sample augmentation seeds (:mod:`training.data.sample_seed`) and the loader's own
    generator, when it has one, yields the epoch's worker base seed.
    """
    sources = (
        getattr(loader, "sampler", None),
        getattr(loader, "batch_sampler", None),
        getattr(loader, "dataset", None),
    )
    for source in sources:
        set_epoch = getattr(source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(DEFAULT_SAMPLER_SEED + int(epoch))


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
