"""Shard-window epoch order and block-wise curriculum subsets (audit 7.1.3)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml
from torch.utils.data import Dataset

from data.dataset import MultiTaskDataset
from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar
from training.data.loaders import (
    LoaderPlacement,
    TrainLoaderOptions,
    WorkerOverrides,
    build_train_loader,
)
from training.data.locality import (
    LocalityConfig,
    ShardWindow,
    block_fraction_subset,
    shard_window_order,
)
from training.data.resumable_sampler import (
    ResumableBatchSampler,
    resume_start_batch,
    set_loader_epoch,
)

SHARDS = 8
PER_SHARD = 10
KEYS = [f"shard-{i // PER_SHARD}" for i in range(SHARDS * PER_SHARD)]


def _shard(index: int) -> str:
    return KEYS[index]


def _windowed(window: int) -> ResumableBatchSampler:
    sampler = ResumableBatchSampler(len(KEYS), 5)
    sampler.set_shard_window(ShardWindow(tuple(KEYS), window) if window else None)
    return sampler


def _order(sampler: ResumableBatchSampler) -> list[int]:
    return [i for batch in sampler for i in batch]


def test_config_defaults_off_and_validates() -> None:
    """Missing sections mean off; bad values are rejected."""
    assert LocalityConfig.from_config(None) == LocalityConfig(0, 0)
    assert (
        LocalityConfig.from_config({"data": {"locality": {"window_shards": 4}}}).window_shards == 4
    )
    with pytest.raises(ValueError):
        LocalityConfig.from_config({"data": {"locality": {"window_shards": -1}}})
    with pytest.raises(TypeError):
        LocalityConfig.from_config({"data": {"locality": {"fraction_block_size": "8"}}})
    with pytest.raises(TypeError):
        LocalityConfig.from_config({"data": {"locality": [1]}})


def test_window_order_is_a_permutation_reading_few_shards_at_a_time() -> None:
    """Each window of the order holds samples of at most ``window`` shards."""
    order = shard_window_order(KEYS, 2, seed=7)
    assert sorted(order) == list(range(len(KEYS)))
    window_len = 2 * PER_SHARD
    for start in range(0, len(order), window_len):
        assert len({_shard(i) for i in order[start : start + window_len]}) == 2
    assert order == shard_window_order(KEYS, 2, seed=7)
    assert order != shard_window_order(KEYS, 2, seed=8)
    with pytest.raises(ValueError):
        shard_window_order(KEYS, 0, seed=7)


def test_sampler_window_order_is_epoch_seeded_and_resumable() -> None:
    """Same seed+epoch gives the same order; resuming at K yields batches K.. exactly."""
    first, second = _windowed(2), _windowed(2)
    first.set_epoch(3)
    second.set_epoch(3)
    full = list(first)
    assert full == list(second)
    second.set_epoch(4)
    assert list(second) != full
    first.set_start_batch(5)
    assert list(first) == full[5:]
    switches = sum(_shard(a) != _shard(b) for a, b in itertools.pairwise(_order(first)))
    uniform = ResumableBatchSampler(len(KEYS), 5)
    uniform_switches = sum(_shard(a) != _shard(b) for a, b in itertools.pairwise(_order(uniform)))
    assert switches < uniform_switches


def test_sampler_window_shards_across_ranks_without_overlap() -> None:
    """DDP ranks split the window order and together cover every sample once."""
    seen: list[int] = []
    for rank in range(2):
        sampler = _windowed(2)
        sampler.rank, sampler.world_size = rank, 2
        sampler.set_epoch(1)
        seen.extend(_order(sampler))
    assert sorted(seen) == list(range(len(KEYS)))


def test_sampler_rejects_mismatched_keys_and_ignores_zero_window() -> None:
    """Keys must cover every sample; no window keeps the uniform permutation."""
    with pytest.raises(ValueError):
        ResumableBatchSampler(3, 1).set_shard_window(ShardWindow(("a",), 2))
    with pytest.raises(ValueError):
        ShardWindow(("a",), 0)
    plain, zero = ResumableBatchSampler(len(KEYS), 5), _windowed(0)
    assert list(plain) == list(zero)


def test_block_subsets_are_nested_and_shard_local() -> None:
    """Larger fractions extend smaller ones; blocks never cross a shard."""
    samples = list(range(len(KEYS)))
    subsets = [block_fraction_subset(samples, KEYS, 5, f) for f in (0.1, 0.25, 0.5, 1.0)]
    assert [len(s) for s in subsets] == [8, 20, 40, 80]
    for smaller, larger in itertools.pairwise(subsets):
        assert larger[: len(smaller)] == smaller
    assert sorted(subsets[-1]) == samples
    first_block = subsets[0][:5]
    assert len({_shard(i) for i in first_block}) == 1
    assert first_block == list(range(first_block[0], first_block[0] + 5))
    assert len({_shard(i) for i in subsets[0]}) < SHARDS
    with pytest.raises(ValueError):
        block_fraction_subset(samples, KEYS[:-1], 5, 0.5)
    with pytest.raises(ValueError):
        block_fraction_subset(samples, KEYS, 0, 0.5)


class _ShardedDataset(Dataset[int]):
    """Index dataset that reports a shard per sample."""

    def __len__(self) -> int:
        return len(KEYS)

    def __getitem__(self, index: int) -> int:
        return index

    def sample_shard_keys(self) -> list[str]:
        """Shard of each index."""
        return list(KEYS)


def test_train_loader_uses_the_configured_window_and_resumes() -> None:
    """``data.locality.window_shards`` reaches the loader's batch sampler."""
    config = {"data": {"locality": {"window_shards": 2}}}
    loader = build_train_loader(
        _ShardedDataset(),
        5,
        options=TrainLoaderOptions(drop_last=False, workers=WorkerOverrides(num_workers=0)),
        placement=LoaderPlacement(config=config),
    )
    sampler = loader.batch_sampler
    assert isinstance(sampler, ResumableBatchSampler) and sampler.shard_window is not None
    assert sampler.shard_window.shards == 2
    set_loader_epoch(loader, 2)
    order = [int(i) for batch in loader for i in batch]
    assert order == shard_window_order(KEYS, 2, seed=sampler.seed + 2)
    start = resume_start_batch(loader, 3)
    assert [int(i) for batch in loader for i in batch] == order[start * 5 :]


def _webdataset(tmp_path: Path, block_size: int) -> MultiTaskDataset:
    model_key = "nafnet_denoising"
    root = Path(__file__).resolve().parents[2]
    registry = yaml.safe_load((root / "unified_models_v2.yaml").read_text(encoding="utf-8"))
    shards = tmp_path / "datasets" / registry[model_key]["datasets"][0] / "shards"
    shards.mkdir(parents=True)
    for shard in range(4):
        members: dict[str, bytes] = {}
        for i in range(6):
            key = f"s{shard}{i:02d}"
            members[f"{key}.webp"] = webp_bytes((40 * shard, 30 * i, 90), size=(16, 16))
            members[f"{key}.target.webp"] = webp_bytes((90, 30 * i, 40 * shard), size=(16, 16))
            members[f"{key}.json"] = json_bytes({"task": "restoration"})
        write_tar(shards / f"shard-{shard:06d}.tar", members)
    config: dict[str, Any] = {
        "unified_models": "unified_models_v2.yaml",
        "paths": {"datasets_root": str(tmp_path / "datasets")},
        "data": {"locality": {"fraction_block_size": block_size}},
    }
    return MultiTaskDataset(config, model_key=model_key, is_train=True, env="local")


def test_webdataset_reports_shards_and_takes_nested_block_fractions(tmp_path: Path) -> None:
    """The dataset maps samples to their tar shards; block fractions stay nested."""
    dataset = _webdataset(tmp_path, block_size=3)
    keys = dataset.shard_keys_for(dataset.catalog.all_samples)
    assert len(set(keys)) == 4
    assert all(shard is not None and shard.endswith(".tar") for _, shard in keys)
    seen: list[list[Any]] = []
    for fraction in (0.25, 0.5, 1.0):
        dataset.update_strategy(fraction=fraction)
        seen.append(list(dataset.catalog.samples))
        assert len(dataset.sample_shard_keys()) == len(dataset)
    assert [len(s) for s in seen] == [6, 12, 24]
    assert seen[1][:6] == seen[0]
    for smaller, larger in itertools.pairwise(seen):
        assert set(smaller) <= set(larger)
    first = dataset.shard_keys_for(seen[0][:3])
    assert len(set(first)) == 1
