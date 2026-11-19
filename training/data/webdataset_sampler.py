"""Deterministic distributed shard sampler for WebDataset archives.

Ensures deterministic, epoch-seeded permutation of dataset shards across
multiple distributed ranks, avoiding redundant reads and data race conditions
while providing fully reproducible training resumes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class DeterministicShardSampler(Sampler[int]):
    """Deterministic shard sampler partitioning shards across distributed ranks per epoch.

    Args:
        num_shards: Total number of shards in the WebDataset archive collection.
        num_replicas: Number of distributed ranks (e.g. world_size).
        rank: Current distributed rank ID.
        seed: Master random seed for reproducible permutations.
        drop_last: Whether to drop remaining shards that cannot be evenly divided.
    """

    def __init__(
        self,
        num_shards: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas {num_replicas}")

        self.num_shards = max(0, num_shards)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if self.drop_last:
            self.num_samples = self.num_shards // self.num_replicas
        else:
            self.num_samples = (
                math.ceil((self.num_shards - self.rank) / self.num_replicas)
                if self.num_shards > self.rank
                else 0
            )

    @property
    def total_size(self) -> int:
        """Total number of distributed samples across all replicas."""
        return (self.num_shards // self.num_replicas) * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for epoch-seeded deterministic permutation."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield shard indices for current rank under epoch seed."""
        if self.num_shards == 0:
            return iter([])

        # Deterministically permute shard indices
        g = random.Random(self.seed + self.epoch * 10007)
        indices = list(range(self.num_shards))
        g.shuffle(indices)

        if self.drop_last:
            # Subsample to total_size
            indices = indices[: self.total_size]
            sub_indices = indices[self.rank : self.total_size : self.num_replicas]
        else:
            # If not drop_last, partition without duplicating within one epoch
            # Rank gets stride across original indices
            sub_indices = indices[self.rank :: self.num_replicas]

        return iter(sub_indices)

    def __len__(self) -> int:
        return self.num_samples
