"""Publication-grade FID invariant and FID consistency (review 5.2 and D-13).

A recording stand-in for TorchMetrics' ``FrechetInceptionDistance`` checks that validation
builds it with 2048 Inception features, feeds real and fake images in pairs of at most 2,
and drops the FID of a pass whose real and fake features no longer cover the same images.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch

from training.training import val_metrics
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import ValBatch, update_image
from training.training.val_summary import summarize


class _RecordingFid:
    """Records constructor arguments and every ``update`` call."""

    instances: list[_RecordingFid] = []
    fail_fake_update = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updates: list[tuple[bool, int]] = []
        _RecordingFid.instances.append(self)

    def to(self, _device: Any) -> _RecordingFid:
        """Device move (no-op)."""
        return self

    def eval(self) -> _RecordingFid:
        """Eval mode (no-op)."""
        return self

    def update(self, images: torch.Tensor, real: bool) -> None:
        """Record the chunk size and the real/fake flag; optionally fail fake updates."""
        if not real and _RecordingFid.fail_fake_update:
            raise RuntimeError("simulated CUDA OOM")
        self.updates.append((real, int(images.size(0))))

    def compute(self) -> torch.Tensor:
        """A fixed FID value."""
        return torch.tensor(12.5)


def _batch(images: int) -> ValBatch:
    pred = torch.rand(images, 3, 8, 8)
    return ValBatch(
        preds=pred,
        targets=pred.clone(),
        sota_cfg={"fid": 10.0},
        dataset_type="restoration",
        device=torch.device("cpu"),
        governor=None,
    )


def _summary(acc: ValAccumulators) -> dict[str, float]:
    ctx = SimpleNamespace(model_name="nafnet_denoising", model=object())
    return summarize(ctx, acc, step_count=1, elapsed=0.0)


class TestFidInvariants(unittest.TestCase):
    """FID uses 2048 features, chunks of at most 2, and is dropped when inconsistent."""

    def setUp(self) -> None:
        _RecordingFid.instances = []
        _RecordingFid.fail_fake_update = False
        patcher = mock.patch.object(val_metrics, "FrechetInceptionDistance", _RecordingFid)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_feature_size_and_pairwise_chunks(self) -> None:
        """2048 features; real/fake updates alternate in chunks of at most 2 images."""
        acc = ValAccumulators()
        update_image(acc, _batch(5))
        fid = _RecordingFid.instances[0]
        self.assertEqual(fid.kwargs["feature"], 2048)
        self.assertEqual(val_metrics.FID_CHUNK_SIZE, 2)
        self.assertEqual([real for real, _ in fid.updates], [True, False] * 3)
        self.assertTrue(all(size <= 2 for _, size in fid.updates))
        self.assertEqual(sum(size for real, size in fid.updates if real), 5)
        self.assertEqual(acc.image.fid_batches_accumulated, 1)
        self.assertEqual(_summary(acc)["fid"], 12.5)

    def test_half_pair_failure_drops_fid(self) -> None:
        """A fake update that fails after its real half marks the pass and FID is not reported."""
        acc = ValAccumulators()
        update_image(acc, _batch(2))
        _RecordingFid.fail_fake_update = True
        with self.assertLogs("lemtrain.training.validation", level="WARNING"):
            update_image(acc, _batch(2))
        self.assertTrue(acc.image.fid_incomplete)
        self.assertEqual(acc.image.fid_batches_accumulated, 1)
        self.assertNotIn("fid", _summary(acc))

    def test_incomplete_flag_is_not_checkpointed(self) -> None:
        """The flag is per pass: progress snapshots neither store nor restore it."""
        acc = ValAccumulators()
        acc.image.fid_incomplete = True
        snapshot = acc.snapshot()
        self.assertNotIn("fid_incomplete", snapshot)
        fresh = ValAccumulators()
        fresh.restore(snapshot)
        self.assertFalse(fresh.image.fid_incomplete)


if __name__ == "__main__":
    unittest.main()
