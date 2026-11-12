"""Quality score: ROBOTS restoration formula, weighted targets, scale migration.

Audit 3.2.4 (a missing target adds no phantom score) and 7.2.1 (an apples-to-apples
score on every epoch, FID kept out of restoration scores).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from training.context_resume import reconcile_metrics_csv
from training.governance.epoch_audit import EpochSignals
from training.governance.governor import SmartTrainingGovernor
from training.governance.quality_score import (
    QUALITY_SCHEME_KEY,
    RESTORATION_SCHEME,
    anchor_is_comparable,
    csv_row_quality,
    score_metrics,
    target_quality_score,
)
from training.telemetry import TelemetryEngine

RESTORATION_SOTA = {"psnr": 32.0, "ssim": 0.92, "lpips": 0.08, "fid": 20.0}
WEIGHTED_SOTA = {"psnr": 30.0, "ssim": 0.9, "fid": 20.0}


def _score(metrics: dict[str, float], targets: dict[str, float], task: str) -> float:
    score, collapsed = TelemetryEngine(".").compute_quality_score(metrics, targets, task)
    assert not collapsed
    return score


def test_missing_target_adds_no_phantom_score() -> None:
    """Weighted tasks: an unmeasured FID contributes nothing instead of a perfect +100."""
    without_fid = _score({"psnr": 28.0, "ssim": 0.85}, WEIGHTED_SOTA, "image")
    with_fid = _score({"psnr": 28.0, "ssim": 0.85, "fid": 35.0}, WEIGHTED_SOTA, "image")
    assert with_fid - without_fid == pytest.approx(100.0 - 35.0)


def test_measured_zero_still_counts() -> None:
    """A measured value of 0.0 is a measurement, not a missing metric."""
    zero = _score({"psnr": 28.0, "ssim": 0.85, "fid": 0.0}, WEIGHTED_SOTA, "image")
    assert zero > _score({"psnr": 28.0, "ssim": 0.85}, WEIGHTED_SOTA, "image")


@pytest.mark.parametrize("task", ["restoration", "enhancement", "face"])
def test_restoration_score_is_the_robots_formula(task: str) -> None:
    """PSNR + 20*SSIM - 20*LPIPS, the same with or without an FID reading."""
    metrics = {"psnr": 30.0, "ssim": 0.9, "lpips": 0.1}
    expected = 30.0 + 20 * 0.9 - 20 * 0.1
    assert _score(metrics, RESTORATION_SOTA, task) == pytest.approx(expected)
    assert _score({**metrics, "fid": 35.0}, RESTORATION_SOTA, task) == pytest.approx(expected)


def test_target_uses_the_epoch_formula() -> None:
    """The curriculum target is the epoch score of the SOTA targets."""
    expected = 32.0 + 20 * 0.92 - 20 * 0.08
    assert target_quality_score(RESTORATION_SOTA, "restoration") == pytest.approx(expected)
    assert score_metrics(RESTORATION_SOTA, RESTORATION_SOTA, "restoration") == pytest.approx(
        expected
    )


def test_csv_rows_are_rescored_on_the_current_scale() -> None:
    """Old restoration rows are rescored from their metric columns."""
    old_row = {"PSNR": "30.0", "SSIM": "0.9", "LPIPS": "0.1", "FID": "", "Quality_Score": "372"}
    assert csv_row_quality(old_row, "restoration") == pytest.approx(46.0)
    assert csv_row_quality({"Quality_Score": "81.5"}, "quality") == pytest.approx(81.5)
    assert csv_row_quality({"Quality_Score": ""}, "quality") is None


def test_anchor_comparability() -> None:
    """Untagged anchors match weighted tasks only; tagged ones match their scheme."""
    assert not anchor_is_comparable(None, "restoration")
    assert anchor_is_comparable(RESTORATION_SCHEME, "restoration")
    assert anchor_is_comparable(None, "quality")
    assert not anchor_is_comparable(RESTORATION_SCHEME, "quality")


def _governor(task: str) -> SmartTrainingGovernor:
    return SmartTrainingGovernor(
        {"name": "qs_model", "dataset_type": task, "input_size": 256, "batch_size": 8}
    )


def test_old_scale_best_quality_is_not_reused(caplog: pytest.LogCaptureFixture) -> None:
    """A restoration checkpoint from the weighted scale drops its best and history."""
    gov = _governor("restoration")
    fresh_best = gov.best_quality
    state = _governor("restoration").get_state()
    state.pop(QUALITY_SCHEME_KEY)
    state.update(best_quality=372.0, history=[(370.0, None, None)])
    with caplog.at_level(logging.WARNING, logger="lemtrain.governance.governor"):
        gov.load_state(state)
    assert gov.best_quality == fresh_best
    assert not gov.history
    assert "another scheme" in caplog.text


def test_tagged_best_quality_round_trips() -> None:
    """A state written with the current scheme restores its best quality."""
    source = _governor("restoration")
    source.best_quality = 47.5
    gov = _governor("restoration")
    gov.load_state(source.get_state())
    assert gov.best_quality == pytest.approx(47.5)


def test_untagged_weighted_best_quality_is_kept() -> None:
    """Weighted-score tasks did not change scale, so untagged states still restore."""
    state = _governor("quality").get_state()
    state.pop(QUALITY_SCHEME_KEY)
    state["best_quality"] = 88.0
    gov = _governor("quality")
    gov.load_state(state)
    assert gov.best_quality == pytest.approx(88.0)


def test_metrics_csv_reconcile_uses_the_current_scale(tmp_path: Path) -> None:
    """``reconcile_metrics_csv`` lifts the best quality to the rescored CSV maximum."""
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "Epoch,PSNR,SSIM,LPIPS,FID,Quality_Score,Res,Data\n"
        "1,28.0,0.85,0.2,,320.0,256,0.2\n"
        "2,30.0,0.9,0.1,25.0,440.0,256,0.2\n",
        encoding="utf-8",
    )
    gov = _governor("restoration")
    gov.best_quality = 0.0
    reconcile_metrics_csv(gov, csv_path, 0.2)
    assert gov.best_quality == pytest.approx(46.0)


def test_target_score_ignores_targets_not_measured_this_epoch() -> None:
    """Only the targets the epoch measured enter the aim (R2-T7)."""
    targets = {"plcc": 0.9, "srcc": 0.9, "rank_margin": 0.05}
    full = target_quality_score(targets, "quality")
    partial = target_quality_score(targets, "quality", {"plcc": 0.5, "srcc": 0.4})
    assert partial == pytest.approx(target_quality_score({"plcc": 0.9, "srcc": 0.9}, "quality"))
    assert partial < full
    nan_measured = {"plcc": 0.5, "srcc": float("nan"), "rank_margin": None}
    assert target_quality_score(targets, "quality", nan_measured) == pytest.approx(
        target_quality_score({"plcc": 0.9}, "quality")
    )
    assert target_quality_score(targets, "quality", {}) == 1.0


def test_governor_audit_aims_at_the_measured_targets() -> None:
    """The governor's target score follows the epoch's measured metrics."""
    governor = SmartTrainingGovernor(
        model_info={
            "dataset_type": "quality",
            "sota_targets": {"plcc": 0.9, "srcc": 0.9, "rank_margin": 0.05},
        },
        config={},
    )
    governor.audit_epoch(
        EpochSignals(
            current_quality=50.0,
            current_loss=1.0,
            train_loss=1.0,
            metrics_dict={"plcc": 0.5, "srcc": 0.5},
        ),
        epochs_no_improve=0,
    )
    assert governor.target_quality_score == pytest.approx(
        target_quality_score({"plcc": 0.9, "srcc": 0.9}, "quality")
    )
