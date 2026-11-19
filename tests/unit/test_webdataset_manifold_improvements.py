"""Unit tests for WebDataset manifold improvements in Prompt 17."""

from __future__ import annotations

import torch
from torch import nn

from training.data.degrade import DegradationCurriculum
from training.data.webdataset_sampler import DeterministicShardSampler
from training.losses import CombinedLoss, _masked_spatial_l1_loss
from training.training.ema import ModelEMA
from training.training.image_throughput import ImageThroughputTracker


class _TinyModel(nn.Module):
    """Minimal visual module for ModelEMA testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv and batch norm."""
        return self.bn(self.conv(x))


def test_model_ema_smoothing() -> None:
    """Verify ModelEMA tracks parameters, applies shadow weights, and restores."""
    model = _TinyModel()
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)

    # Initial weights should match
    with torch.no_grad():
        for p_online, p_shadow in zip(model.parameters(), ema.module.parameters()):
            assert torch.allclose(p_online, p_shadow)

    # Perturb online model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema.update(model)

    # Shadow should be between original and +1.0
    with torch.no_grad():
        for p_online, p_shadow in zip(model.parameters(), ema.module.parameters()):
            assert not torch.allclose(p_online, p_shadow)
            assert torch.all(p_shadow < p_online)

    # Test apply_shadow and restore
    orig_online_weight = model.conv.weight.clone()
    ema.apply_shadow(model)
    assert torch.allclose(model.conv.weight, ema.module.conv.weight)

    ema.restore(model)
    assert torch.allclose(model.conv.weight, orig_online_weight)

    # Test state_dict round-trip
    state = ema.state_dict()
    new_ema = ModelEMA(model, decay=0.5)
    new_ema.load_state_dict(state)
    assert new_ema.step == ema.step
    assert new_ema.decay == ema.decay
    assert torch.allclose(new_ema.module.conv.weight, ema.module.conv.weight)


def test_degradation_curriculum_monotonicity() -> None:
    """Verify DegradationCurriculum ramps difficulty monotonically with epoch."""
    curriculum = DegradationCurriculum(total_epochs=50, start_intensity=0.1, max_intensity=1.0)

    d0 = curriculum.get_difficulty(0)
    d25 = curriculum.get_difficulty(25)
    d50 = curriculum.get_difficulty(50)

    assert d0 < d25 < d50
    assert abs(d0 - 0.1) < 1e-5
    assert abs(d50 - 1.0) < 1e-5

    dummy_img = torch.rand(3, 64, 64)
    out_img, meta = curriculum(dummy_img, current_epoch=25, rung_resolution=128)
    assert out_img.shape == dummy_img.shape
    assert 0.0 <= out_img.min() <= out_img.max() <= 1.0
    assert meta["epoch"] == 25
    assert meta["rung_resolution"] == 128


def test_masked_spatial_l1_loss() -> None:
    """Verify mask-aware loss weights regions according to ROI mask."""
    pred = torch.zeros(1, 1, 10, 10)
    target = torch.ones(1, 1, 10, 10)

    # Without mask: mean difference is 1.0
    unmasked_loss = _masked_spatial_l1_loss(pred, target, mask=None)
    assert abs(unmasked_loss.item() - 1.0) < 1e-5

    # Mask covering half the image
    mask = torch.zeros(1, 1, 10, 10)
    mask[:, :, :5, :] = 1.0

    # With uniform error, weighted sum over weighted elements is still 1.0
    loss_uniform = _masked_spatial_l1_loss(pred, target, mask=mask, mask_weight=3.0)
    assert abs(loss_uniform.item() - 1.0) < 1e-5

    # Target has error ONLY in masked area
    target_masked_only = torch.zeros(1, 1, 10, 10)
    target_masked_only[:, :, :5, :] = 1.0

    loss_weighted = _masked_spatial_l1_loss(pred, target_masked_only, mask=mask, mask_weight=5.0)
    # Masked region has weight 5.0, unmasked region has weight 1.0
    # 50 pixels * 5.0 = 250 error sum, total weight = 50 * 5.0 + 50 * 1.0 = 300
    # Expected weighted loss = 250 / 300 = 0.8333...
    assert abs(loss_weighted.item() - (250.0 / 300.0)) < 1e-4

    # CombinedLoss integration test with mask in target dict
    combined_loss = CombinedLoss(task_type="restoration", use_perc=False)
    loss_val = combined_loss(pred, {"image": target_masked_only, "mask": mask})
    assert loss_val.item() > 0.0


def test_deterministic_shard_sampler() -> None:
    """Verify DeterministicShardSampler partition consistency and epoch seed variation."""
    num_shards = 17
    num_replicas = 4

    sampler_rank0 = DeterministicShardSampler(
        num_shards=num_shards, num_replicas=num_replicas, rank=0, seed=123
    )
    sampler_rank1 = DeterministicShardSampler(
        num_shards=num_shards, num_replicas=num_replicas, rank=1, seed=123
    )

    sampler_rank0.set_epoch(0)
    sampler_rank1.set_epoch(0)

    shards_r0 = list(sampler_rank0)
    shards_r1 = list(sampler_rank1)

    assert len(shards_r0) == len(sampler_rank0)
    assert len(shards_r1) == len(sampler_rank1)
    # Ranks should receive disjoint shards
    assert set(shards_r0).isdisjoint(set(shards_r1))

    # Same rank with same epoch must produce identical shard sequence
    sampler_rank0_repeat = DeterministicShardSampler(
        num_shards=num_shards, num_replicas=num_replicas, rank=0, seed=123
    )
    sampler_rank0_repeat.set_epoch(0)
    assert list(sampler_rank0_repeat) == shards_r0

    # Changing epoch should permute the shards
    sampler_rank0.set_epoch(1)
    shards_r0_epoch1 = list(sampler_rank0)
    assert shards_r0_epoch1 != shards_r0


def test_image_throughput_tracker() -> None:
    """Verify ImageThroughputTracker registers steps and aggregates metrics."""
    tracker = ImageThroughputTracker(device=torch.device("cpu"))
    reading1 = tracker.record_step(num_images=16)
    assert reading1.total_images == 16
    assert reading1.images_per_sec > 0.0

    reading2 = tracker.record_step(num_images=32)
    assert reading2.total_images == 48

    summary = tracker.summary()
    assert summary["total_images"] == 48.0
    assert summary["avg_images_per_sec"] > 0.0
