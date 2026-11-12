"""Governor LR multipliers compose with the scheduler's decay (audit 3.2.5)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from training.training.lr_scaling import APPLIED_MULT_KEY, GROUP_ROLE_KEY, scale_learning_rates

BASE_LR = 0.1


def _optimizer() -> torch.optim.Optimizer:
    backbone, head = nn.Linear(2, 2), nn.Linear(2, 1)
    return torch.optim.SGD(
        [
            {"params": backbone.parameters()},
            {"params": head.parameters(), GROUP_ROLE_KEY: "head"},
        ],
        lr=BASE_LR,
    )


def _step(optimizer: torch.optim.Optimizer, scheduler, epochs: int) -> None:
    for _ in range(epochs):
        optimizer.step()
        scheduler.step()


def test_multiplier_keeps_cosine_decay() -> None:
    """A cooled run follows the same cosine curve, scaled by the multiplier."""
    reference, cooled = _optimizer(), _optimizer()
    ref_sched = CosineAnnealingLR(reference, T_max=10)
    cooled_sched = CosineAnnealingLR(cooled, T_max=10)
    _step(reference, ref_sched, 3)
    _step(cooled, cooled_sched, 3)

    scale_learning_rates(cooled, cooled_sched, 0.5, 0.5)
    assert cooled.param_groups[0]["lr"] == pytest.approx(reference.param_groups[0]["lr"] * 0.5)

    _step(reference, ref_sched, 2)
    _step(cooled, cooled_sched, 2)
    assert cooled.param_groups[0]["lr"] == pytest.approx(reference.param_groups[0]["lr"] * 0.5)


def test_multiplier_survives_closed_form_scheduler() -> None:
    """``LambdaLR`` recomputes from ``base_lrs``; the multiplier is kept there too."""
    optimizer = _optimizer()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (epoch + 1))
    _step(optimizer, scheduler, 1)
    scale_learning_rates(optimizer, scheduler, 0.5, 0.5)
    _step(optimizer, scheduler, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR * 0.5 / 3)


def test_ratio_is_applied_once_and_reverts() -> None:
    """Re-applying the same multiplier is a no-op; returning to 1.0 restores the rate."""
    optimizer = _optimizer()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scale_learning_rates(optimizer, scheduler, 0.5, 0.5)
    scale_learning_rates(optimizer, scheduler, 0.5, 0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR * 0.5)
    assert optimizer.param_groups[0][APPLIED_MULT_KEY] == 0.5
    scale_learning_rates(optimizer, scheduler, 1.0, 1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR)
    assert scheduler.base_lrs[0] == pytest.approx(BASE_LR)


def test_head_multiplier_only_scales_head_groups() -> None:
    """``head_lr_multiplier`` applies to groups tagged ``head``; the rest follow the backbone."""
    optimizer = _optimizer()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scale_learning_rates(optimizer, scheduler, 0.5, 1.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(BASE_LR * 0.5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(BASE_LR * 1.5)


def test_applied_multiplier_round_trips_optimizer_state() -> None:
    """The applied multiplier is part of the optimizer state saved in checkpoints."""
    optimizer = _optimizer()
    scale_learning_rates(optimizer, None, 0.25, 0.25)
    restored = _optimizer()
    restored.load_state_dict(optimizer.state_dict())
    assert restored.param_groups[0][APPLIED_MULT_KEY] == 0.25
    scale_learning_rates(restored, None, 1.0, 1.0)
    assert restored.param_groups[0]["lr"] == pytest.approx(BASE_LR)
