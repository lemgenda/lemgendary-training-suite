"""Per-sample augmentation seeds that a mid-epoch resume reproduces exactly.

Sample augmentation (random crops and flips, synthesized degradation, film grain) used to
draw its seed with ``random.randint`` inside the loader workers, so a resumed epoch got
different augmentations than the interrupted run would have. Each sample's seed is now a
pure function of ``(base seed, epoch, dataset index)``: it does not depend on which worker,
DDP rank or batch position fetches the sample, so it is the same under
``ResumableBatchSampler`` resumes, DDP sharding and shard-locality ordering.

The epoch lives in shared memory so persistent workers see :meth:`AugmentationSeed.set_epoch`
without being re-spawned. :func:`seeded_sample_rngs` seeds Python ``random``, NumPy and
torch for one sample and restores the previous states afterwards, so loading in the main
process (``num_workers=0``) never perturbs the training loop's generators.
"""

from __future__ import annotations

import multiprocessing
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from training.data.resumable_sampler import DEFAULT_SAMPLER_SEED
from training.utils.rng_state import capture_host_rng, restore_host_rng


def derive_sample_seed(base_seed: int, epoch: int, index: int) -> int:
    """The 32-bit augmentation seed of dataset sample ``index`` in ``epoch``."""
    sequence = np.random.SeedSequence((int(base_seed), int(epoch), int(index)))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class AugmentationSeed:
    """Epoch-seeded per-sample augmentation seeds shared with the loader workers."""

    def __init__(self, base_seed: int = DEFAULT_SAMPLER_SEED) -> None:
        self.base_seed = int(base_seed)
        # Shared memory: persistent workers read the epoch the main process sets.
        self._epoch = multiprocessing.Value("q", 0)

    @property
    def epoch(self) -> int:
        """The epoch whose seeds :meth:`sample_seed` derives."""
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """Derive the seeds of ``epoch`` from now on (all workers included)."""
        self._epoch.value = int(epoch)

    def sample_seed(self, index: int) -> int:
        """The augmentation seed of dataset sample ``index`` in the current epoch."""
        return derive_sample_seed(self.base_seed, self.epoch, index)


@contextmanager
def seeded_sample_rngs(seed: int) -> Iterator[None]:
    """Seed ``random``, NumPy and torch CPU with ``seed``; restore their states on exit."""
    saved = capture_host_rng()
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        restore_host_rng(saved)
