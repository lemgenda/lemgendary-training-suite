"""Unit and regression tests for Training Pathology audit fixes (Prompt 15).

Covers:
- Pathology 16: Post-training target audit and diagnostic guidance.
- Pathology 23: Polarity inversion shield detecting negative SRCC/PLCC.
- Pathology 59: Rank margin tightening during targeted deficit mitigation.
- Pathology 62: Terminal progress resume guard advancing at >= 99.9%.
- Pathology 66: Unlabelled batch warning in classification loss.
"""

import logging

import pytest
import torch
from torch import nn

from training.checkpoint.resume import _progress, parse_resume_state
from training.governance import SmartTrainingGovernor
from training.governance.epoch_audit import EpochSignals
from training.losses import _classification_loss
from training.target_audit import audit_final_targets, format_post_training_audit


def test_post_training_target_audit_formatting():
    """Verify Pathology 16 post-training target audit table and diagnostics."""
    achieved = {"plcc": 0.925, "srcc": 0.880, "test_emd": 0.045}
    targets = {"plcc": 0.910, "srcc": 0.910, "emd": 0.050}

    results = audit_final_targets(achieved, targets)
    assert results["plcc"]["met"] is True
    assert results["srcc"]["met"] is False
    assert results["srcc"]["gap"] == pytest.approx(0.030)
    assert results["emd"]["met"] is True

    report = format_post_training_audit(
        model_name="nima_technical",
        final_epoch=150,
        status="completed",
        achieved_metrics=achieved,
        sota_targets=targets,
    )
    assert "POST-TRAINING TARGET AUDIT: nima_technical (Epoch 150)" in report
    assert "[PASS]" in report
    assert "[DEFICIT]" in report
    assert "Escalate to Kaggle Cloud Hub" in report


def test_polarity_inversion_shield():
    """Verify Pathology 23 triggers collapse when correlation is negative."""
    governor = SmartTrainingGovernor(
        model_info={"name": "test_scorer", "task_type": "quality"},
        config={"governor": {"enabled": True}},
    )

    signals_inverted = EpochSignals(
        current_quality=50.0,
        current_loss=0.20,
        train_loss=0.18,
        plcc=-0.05,
        srcc=-0.12,
        metrics_dict={"plcc": -0.05, "srcc": -0.12},
    )

    audit_res = governor.audit_epoch(signals_inverted, epochs_no_improve=0)
    status_msg = str(audit_res[7])
    assert "NPP FAILURE" in status_msg or "RECOIL" in status_msg


def test_terminal_progress_guard_advances_epoch():
    """Verify Pathology 62 treats >= 99.9% progress as finished epoch."""
    payload_completed = {
        "epoch": 10,
        "progress_metadata": {
            "epoch": 10,
            "phase": "train",
            "step": 9995,
            "total_steps": 10000,
        },
    }
    phase, is_intra, step = _progress(payload_completed, 0)
    assert phase == "train"
    assert is_intra is False
    assert step == 9995

    resume_state = parse_resume_state(payload_completed)
    assert resume_state.is_intra_epoch is False

    payload_mid_epoch = {
        "epoch": 10,
        "progress_metadata": {
            "epoch": 10,
            "phase": "train",
            "step": 5000,
            "total_steps": 10000,
        },
    }
    _phase_mid, is_intra_mid, step_mid = _progress(payload_mid_epoch, 0)
    assert is_intra_mid is True
    assert step_mid == 5000


def test_unlabelled_batch_loss_warning(caplog):
    """Verify Pathology 66 logs diagnostic warning on completely unlabelled batch."""
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    pred = torch.randn(4, 5, requires_grad=True)
    target = torch.full((4,), 255, dtype=torch.long)

    with caplog.at_level(logging.WARNING):
        loss = _classification_loss(criterion, pred, target)

    assert float(loss.item()) == 0.0
    assert "[ZERO LOSS GUARD]" in caplog.text


def test_rank_margin_tightening_in_targeted_optimization():
    """Verify Pathology 59 tightens rank margin when SRCC/PLCC is lagging."""
    governor = SmartTrainingGovernor(
        model_info={
            "name": "test_nima",
            "task_type": "quality",
            "sample_fraction": 1.0,
            "res_ladder": [512],
            "sota_targets": {"srcc": 0.91, "plcc": 0.91},
        },
        config={"governor": {"enabled": True, "res_ladder": [512]}},
    )
    governor.curriculum.current_res = 512
    governor.curriculum.current_fraction = 1.0

    initial_margin = governor.current_rank_margin
    signals = EpochSignals(
        current_quality=50.0,
        current_loss=0.10,
        train_loss=0.10,
        plcc=0.92,
        srcc=0.75,
        metrics_dict={"srcc": 0.75, "plcc": 0.92},
    )
    audit_res = governor.audit_epoch(signals, epochs_no_improve=6)
    msg = str(audit_res[7])

    assert governor.current_rank_margin < initial_margin
    assert "Margin:" in msg
