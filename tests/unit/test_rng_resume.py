"""Generator state in checkpoints and augmentation that a mid-epoch resume reproduces (R2-T8).

An epoch is interrupted after ``INTERRUPT_AT`` batches, the generator record is taken the
way a progress checkpoint takes it, the process state is scrambled and a fresh dataset and
loader resume at that batch. The remaining batches' augmented tensors, and the training
loop's own draws after each batch, must equal those of an uninterrupted run.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from data.dataset import MultiTaskDataset
from tests.fixtures.webdataset import json_bytes, write_tar
from training.data.loaders import TrainLoaderOptions, WorkerOverrides, build_train_loader
from training.data.resumable_sampler import resume_start_batch, set_loader_epoch
from training.data.sample_seed import AugmentationSeed, derive_sample_seed, seeded_sample_rngs
from training.utils import rng_state
from training.utils.rng_state import (
    PAYLOAD_KEY,
    DeviceGenerators,
    capture_rng_state,
    restore_rng_state,
)

MODEL_KEY = "nafnet_denoising"
SAMPLES = 24
BATCH = 4
EPOCH = 3
INTERRUPT_AT = 2
CROP = 16

Batches = list[tuple[torch.Tensor, torch.Tensor]]


def _noise_webp(seed: int) -> bytes:
    """A lossless WebP of random pixels (so crops and degradations differ visibly)."""
    pixels = np.random.default_rng(seed).integers(0, 256, size=(24, 40, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


def _write_shard(root: Path) -> dict[str, Any]:
    """Unpaired restoration samples: each input is degraded on the fly (random)."""
    members: dict[str, bytes] = {}
    for i in range(SAMPLES):
        members[f"s{i:03d}.webp"] = _noise_webp(i)
        members[f"s{i:03d}.json"] = json_bytes({"task": "restoration"})
    data_root = root / "datasets"
    registry_path = Path(__file__).resolve().parents[2] / "unified_models_v2.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    shards = data_root / registry[MODEL_KEY]["datasets"][0] / "shards"
    shards.mkdir(parents=True)
    write_tar(shards / "shard-000000.tar", members)
    return {"unified_models": "unified_models_v2.yaml", "paths": {"datasets_root": str(data_root)}}


def _loader(config: dict[str, Any], workers: int):
    dataset = MultiTaskDataset(config, model_key=MODEL_KEY, is_train=True, env="local")
    dataset.update_strategy(size=CROP)
    options = TrainLoaderOptions(
        workers=WorkerOverrides(num_workers=workers, persistent_workers=workers > 0)
    )
    return build_train_loader(dataset, BATCH, options)


def _step(batch: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Keep the batch tensors and draw like a training step would (dropout, noise)."""
    inputs, targets = batch[0], batch[1]
    return (inputs.clone(), targets.clone()), torch.rand(3)


def _uninterrupted(config: dict[str, Any], workers: int) -> tuple[Batches, list[torch.Tensor]]:
    torch.manual_seed(7)
    loader = _loader(config, workers)
    set_loader_epoch(loader, EPOCH)
    batches, draws = [], []
    for batch in loader:
        tensors, draw = _step(batch)
        batches.append(tensors)
        draws.append(draw)
    return batches, draws


def _interrupted_then_resumed(
    config: dict[str, Any], workers: int
) -> tuple[Batches, list[torch.Tensor]]:
    torch.manual_seed(7)
    loader = _loader(config, workers)
    set_loader_epoch(loader, EPOCH)
    payload: dict[str, Any] = {}
    for batch_idx, batch in enumerate(loader):
        _step(batch)
        if batch_idx + 1 == INTERRUPT_AT:
            payload = {PAYLOAD_KEY: capture_rng_state(), "iteration": INTERRUPT_AT}
            break
    del loader

    # A new process: other generator states and a freshly built dataset and loader.
    torch.manual_seed(12345)
    np.random.seed(12345)
    random.seed(12345)
    resumed = _loader(config, workers)
    assert restore_rng_state(payload)
    set_loader_epoch(resumed, EPOCH)
    assert resume_start_batch(resumed, payload["iteration"]) == INTERRUPT_AT
    batches, draws = [], []
    for batch in resumed:
        tensors, draw = _step(batch)
        batches.append(tensors)
        draws.append(draw)
    return batches, draws


def _assert_same(expected: Batches, actual: Batches) -> None:
    assert len(expected) == len(actual)
    for (exp_in, exp_tgt), (act_in, act_tgt) in zip(expected, actual):
        assert torch.equal(exp_in, act_in)
        assert torch.equal(exp_tgt, act_tgt)


@pytest.mark.parametrize("workers", [0, 2])
def test_mid_epoch_resume_reproduces_augmentation_and_loop_draws(
    tmp_path: Path, workers: int
) -> None:
    """The resumed batches and loop draws equal the uninterrupted run's, worker count aside."""
    config = _write_shard(tmp_path)
    full_batches, full_draws = _uninterrupted(config, workers)
    resumed_batches, resumed_draws = _interrupted_then_resumed(config, workers)

    _assert_same(full_batches[INTERRUPT_AT:], resumed_batches)
    for expected, actual in zip(full_draws[INTERRUPT_AT:], resumed_draws):
        assert torch.equal(expected, actual)


def test_augmentation_is_random_across_samples_and_epochs(tmp_path: Path) -> None:
    """Seeds differ per sample and epoch, so augmentation still varies."""
    config = _write_shard(tmp_path)
    dataset = MultiTaskDataset(config, model_key=MODEL_KEY, is_train=True, env="local")
    dataset.update_strategy(size=CROP)
    dataset.set_epoch(1)
    first = dataset[0][0]
    again = dataset[0][0]
    dataset.set_epoch(2)
    next_epoch = dataset[0][0]
    assert torch.equal(first, again)
    assert not torch.equal(first, next_epoch)
    seeds = {derive_sample_seed(42, 1, index) for index in range(SAMPLES)}
    assert len(seeds) == SAMPLES


def test_loading_in_the_main_process_leaves_the_loop_generators_alone() -> None:
    """Seeding one sample restores the Python, NumPy and torch states afterwards."""
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    expected = (torch.rand(2), np.random.rand(2), random.random())
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    with seeded_sample_rngs(AugmentationSeed().sample_seed(5)):
        torch.rand(10)
        np.random.rand(10)
        random.random()
    actual = (torch.rand(2), np.random.rand(2), random.random())
    assert torch.equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
    assert expected[2] == actual[2]


def test_rng_record_loads_weights_only(tmp_path: Path) -> None:
    """The record survives ``torch.load(weights_only=True)`` and restores every generator."""
    path = tmp_path / "progress.pth"
    torch.save({PAYLOAD_KEY: capture_rng_state()}, path)
    expected = (torch.rand(2), np.random.rand(2), random.random())
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert restore_rng_state(loaded)
    actual = (torch.rand(2), np.random.rand(2), random.random())
    assert torch.equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
    assert expected[2] == actual[2]


def test_legacy_checkpoint_without_record_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A checkpoint written before the record existed resumes with fresh generators."""
    with caplog.at_level("INFO", logger="lemtrain.rng_state"):
        assert not restore_rng_state({"epoch": 4})
    assert "no generator state" in caplog.text


def test_malformed_record_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    """An unreadable record keeps the current generators and logs a warning."""
    with caplog.at_level("WARNING", logger="lemtrain.rng_state"):
        assert not restore_rng_state({PAYLOAD_KEY: {"python": [3]}})
    assert "could not be restored" in caplog.text


def test_device_generators_are_captured_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present device backend is recorded and restored per device; an absent one is skipped."""
    store = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
    written: dict[int, torch.Tensor] = {}
    fake = DeviceGenerators(
        name="fake",
        available=lambda: True,
        get_all=lambda: list(store),
        set_one=lambda state, index: written.__setitem__(index, state),
    )
    absent = DeviceGenerators(
        name="absent",
        available=lambda: False,
        get_all=list,
        set_one=lambda state, index: written.__setitem__(-1, state),
    )
    monkeypatch.setattr(rng_state, "DEVICE_GENERATORS", (fake, absent))
    record = capture_rng_state()
    assert [int(t.item()) for t in record["devices"]["fake"]] == [1, 2]
    assert "absent" not in record["devices"]

    record["devices"]["absent"] = [torch.tensor([9], dtype=torch.uint8)]
    assert restore_rng_state({PAYLOAD_KEY: record})
    assert sorted(written) == [0, 1]
