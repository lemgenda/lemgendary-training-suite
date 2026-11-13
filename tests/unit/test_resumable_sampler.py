"""Epoch-seeded, resumable batch sampling (audit 3.3.1)."""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, Dataset

from training.data.loaders import (
    TrainLoaderOptions,
    ValLoaderOptions,
    WorkerOverrides,
    build_train_loader,
    build_val_loader,
)
from training.data.resumable_sampler import (
    ResumableBatchSampler,
    loader_batch_size,
    resume_start_batch,
    set_loader_epoch,
)


class _CountingDataset(Dataset[int]):
    """Returns the index and records every index that was read."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.reads: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        self.reads.append(index)
        return index


def _batches(sampler: ResumableBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


def test_same_epoch_gives_same_permutation() -> None:
    """The permutation depends only on seed and epoch."""
    first, second = ResumableBatchSampler(20, 4), ResumableBatchSampler(20, 4)
    first.set_epoch(3)
    second.set_epoch(3)
    assert _batches(first) == _batches(second)
    second.set_epoch(4)
    assert _batches(first) != _batches(second)


def test_start_batch_resumes_the_interrupted_order_once() -> None:
    """Resuming at K yields exactly the batches K.. of the epoch, then resets."""
    sampler = ResumableBatchSampler(22, 4)
    sampler.set_epoch(1)
    full = _batches(sampler)
    sampler.set_start_batch(3)
    assert _batches(sampler) == full[3:]
    assert _batches(sampler) == full
    assert len(sampler) == len(full) == 6


def test_drop_last_and_order() -> None:
    """Unshuffled batches are in order; ``drop_last`` drops the short tail."""
    sampler = ResumableBatchSampler(10, 4, shuffle=False, drop_last=True)
    assert _batches(sampler) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(sampler) == 2


def test_ranks_share_the_epoch_without_overlap() -> None:
    """Under DDP each rank takes a strided share of the same padded permutation."""
    shards = []
    for rank in range(2):
        sampler = ResumableBatchSampler(9, 2)
        sampler.rank, sampler.world_size = rank, 2
        sampler.set_epoch(5)
        shards.append([i for batch in _batches(sampler) for i in batch])
        assert len(sampler) == 3
    assert len(shards[0]) == len(shards[1]) == 5
    assert set(shards[0]) | set(shards[1]) == set(range(9))


def test_rejects_non_positive_batch_size() -> None:
    """A zero batch size is a configuration error."""
    with pytest.raises(ValueError):
        ResumableBatchSampler(4, 0)


def test_resumed_loader_reads_no_skipped_samples() -> None:
    """Fast-forwarding a loader to batch K reads none of the samples before it."""
    dataset = _CountingDataset(12)
    loader = build_train_loader(
        dataset,
        3,
        options=TrainLoaderOptions(drop_last=False, workers=WorkerOverrides(num_workers=0)),
    )
    set_loader_epoch(loader, 2)
    order = [int(i) for batch in loader for i in batch]
    dataset.reads.clear()

    start = resume_start_batch(loader, 2)
    resumed = list(enumerate(loader, start=start))
    assert start == 2 and resumed[0][0] == 2
    assert [int(i) for _, batch in resumed for i in batch] == order[6:]
    assert sorted(dataset.reads) == sorted(order[6:])
    assert loader_batch_size(loader, 16) == 3


def test_explicit_sampler_falls_back_to_skipping() -> None:
    """A custom sampler keeps the caller-side skip (start 0)."""
    loader = DataLoader(_CountingDataset(4), batch_size=2)
    assert resume_start_batch(loader, 1) == 0
    assert loader_batch_size(loader, 16) == 2


def test_val_loader_is_ordered_and_resumable() -> None:
    """Validation batches keep dataset order and resume without reading earlier batches."""
    loader = build_val_loader(
        _CountingDataset(7), 3, options=ValLoaderOptions(workers=WorkerOverrides(num_workers=0))
    )
    assert [[int(i) for i in b] for b in loader] == [[0, 1, 2], [3, 4, 5], [6]]
    assert resume_start_batch(loader, 2) == 2
    assert [[int(i) for i in b] for b in loader] == [[6]]
