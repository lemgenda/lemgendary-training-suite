"""Quality score of an epoch whose metrics miss a SOTA target (audit 3.2.4)."""

from __future__ import annotations

import pytest

from training.telemetry import TelemetryEngine

SOTA = {"psnr": 30.0, "ssim": 0.9, "fid": 20.0}


def _score(metrics: dict[str, float]) -> float:
    score, collapsed = TelemetryEngine(".").compute_quality_score(metrics, SOTA, "restoration")
    assert not collapsed
    return score


def test_missing_fid_adds_no_phantom_score() -> None:
    """An unmeasured FID contributes nothing instead of a perfect +100."""
    without_fid = _score({"psnr": 28.0, "ssim": 0.85})
    with_fid = _score({"psnr": 28.0, "ssim": 0.85, "fid": 35.0})
    assert with_fid - without_fid == pytest.approx(100.0 - 35.0)


def test_measured_zero_still_counts() -> None:
    """A measured value of 0.0 is a measurement, not a missing metric."""
    assert _score({"psnr": 28.0, "ssim": 0.85, "fid": 0.0}) > _score({"psnr": 28.0, "ssim": 0.85})
