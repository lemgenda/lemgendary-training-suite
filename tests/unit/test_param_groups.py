"""Backbone/head param groups, schedule-preserving head LR and legacy checkpoint load."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from torch import nn

from training.context_resume import TrainState, restore_optimizer_state
from training.training.lr_scaling import GROUP_ROLE_KEY, HEAD_ROLE, scale_learning_rates
from training.training.optimizer import build_optimizer, build_scheduler
from training.training.param_groups import (
    BACKBONE_ROLE,
    build_param_groups,
    legacy_group_sources,
    migrate_legacy_optimizer_state,
    migrate_scheduler_state,
)

_CONFIG: dict[str, Any] = {
    "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.01},
    "scheduler": {"type": "cosine", "min_lr": 0.0},
}


class _Net(nn.Module):
    """Backbone (conv + BatchNorm) and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify ``x``."""
        return self.head(self.pool(self.features(x)).flatten(1))


class _DeclaredHead(_Net):
    """Names its head explicitly, overriding the last-child rule."""

    head_param_prefixes = ("features.1.", "head.")


def _legacy_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """The optimizer layout written by checkpoints before the backbone/head split."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        lowered = name.lower()
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in lowered or "bn" in lowered:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=1e-3,
    )


def _train_steps(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, n: int):
    torch.manual_seed(0)
    for _ in range(n):
        optimizer.zero_grad()
        model(torch.randn(2, 3, 6, 6)).sum().backward()
        optimizer.step()
        scheduler.step()


def _names(model: nn.Module, group: dict[str, Any]) -> set[str]:
    ids = {id(p) for p in group["params"]}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def test_groups_split_backbone_and_head() -> None:
    """Four tagged groups partition the parameters; the last child is the head."""
    model = _Net()
    groups = build_param_groups(model, 0.01)
    roles = [(g[GROUP_ROLE_KEY], g["weight_decay"]) for g in groups]
    assert roles == [
        (BACKBONE_ROLE, 0.01),
        (BACKBONE_ROLE, 0.0),
        (HEAD_ROLE, 0.01),
        (HEAD_ROLE, 0.0),
    ]
    assert _names(model, groups[2]) == {"head.weight"}
    assert _names(model, groups[3]) == {"head.bias"}
    every = [n for g in groups for n in _names(model, g)]
    assert sorted(every) == sorted(n for n, _ in model.named_parameters())


def test_declared_head_prefixes_win() -> None:
    """``head_param_prefixes`` names the head instead of the last child."""
    model = _DeclaredHead()
    head = [g for g in build_param_groups(model, 0.01) if g[GROUP_ROLE_KEY] == HEAD_ROLE]
    assert set().union(*(_names(model, g) for g in head)) == {
        "features.1.weight",
        "features.1.bias",
        "head.weight",
        "head.bias",
    }


def test_single_block_model_has_no_head_group() -> None:
    """A model with one parameterised child keeps the two backbone groups."""
    groups = build_param_groups(nn.Sequential(nn.Linear(3, 2)), 0.01)
    assert [g[GROUP_ROLE_KEY] for g in groups] == [BACKBONE_ROLE, BACKBONE_ROLE]


def test_head_multiplier_keeps_the_schedule() -> None:
    """A head multiplier scales head groups only and survives later scheduler steps."""
    model = _Net()
    optimizer = build_optimizer(model, _CONFIG)
    scheduler = build_scheduler(optimizer, _CONFIG, total_epochs=10)
    _train_steps(model, optimizer, scheduler, 3)
    scale_learning_rates(optimizer, scheduler, backbone_mult=1.0, head_mult=2.0)
    for _ in range(3):
        _train_steps(model, optimizer, scheduler, 1)
        lrs = {(g[GROUP_ROLE_KEY], g["weight_decay"]): g["lr"] for g in optimizer.param_groups}
        for decay in (0.01, 0.0):
            assert lrs[(HEAD_ROLE, decay)] == pytest.approx(2.0 * lrs[(BACKBONE_ROLE, decay)])
    expected_backbone = 1e-3 * 0.5 * (1 + torch.cos(torch.tensor(6 / 10 * torch.pi)).item())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_backbone)


def test_legacy_checkpoint_migrates_state_and_schedule() -> None:
    """A two-group checkpoint loads into the four groups with its moments and LR curve."""
    model = _Net()
    legacy = _legacy_optimizer(model)
    legacy_sched = build_scheduler(legacy, _CONFIG, total_epochs=10)
    _train_steps(model, legacy, legacy_sched, 4)
    saved_opt, saved_sched = legacy.state_dict(), legacy_sched.state_dict()

    optimizer = build_optimizer(model, _CONFIG)
    scheduler = build_scheduler(optimizer, _CONFIG, total_epochs=10)
    migrated = migrate_legacy_optimizer_state(model, optimizer, saved_opt)
    assert migrated is not None and migrated is not saved_opt
    optimizer.load_state_dict(migrated)
    scheduler.load_state_dict(
        migrate_scheduler_state(saved_sched, legacy_group_sources(model, optimizer))
    )

    for param in model.parameters():
        torch.testing.assert_close(
            optimizer.state[param]["exp_avg"], legacy.state[param]["exp_avg"]
        )
        torch.testing.assert_close(optimizer.state[param]["step"], legacy.state[param]["step"])
    assert len(scheduler.base_lrs) == len(optimizer.param_groups) == 4
    legacy_lrs = [g["lr"] for g in legacy.param_groups]
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
        [legacy_lrs[s] for s in legacy_group_sources(model, optimizer)]
    )
    assert [g[GROUP_ROLE_KEY] for g in optimizer.param_groups][-1] == HEAD_ROLE
    _train_steps(model, optimizer, scheduler, 1)
    _train_steps(model, legacy, legacy_sched, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(legacy.param_groups[0]["lr"])


def test_current_layout_is_returned_unchanged() -> None:
    """A state written with the current groups is not touched."""
    model = _Net()
    optimizer = build_optimizer(model, _CONFIG)
    saved = optimizer.state_dict()
    assert migrate_legacy_optimizer_state(model, optimizer, saved) is saved


def test_foreign_state_is_rejected() -> None:
    """A state from another architecture yields ``None`` (caller keeps fresh state)."""
    model = _Net()
    optimizer = build_optimizer(model, _CONFIG)
    other = _legacy_optimizer(nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 1)))
    assert migrate_legacy_optimizer_state(model, optimizer, other.state_dict()) is None


def _train_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> TrainState:
    return TrainState(
        raw_model=model,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        governor=None,
        sota_tracker=None,
    )


def test_resume_restores_legacy_state(caplog: pytest.LogCaptureFixture) -> None:
    """The resume path migrates a legacy state and says so."""
    model = _Net()
    legacy = _legacy_optimizer(model)
    _train_steps(model, legacy, torch.optim.lr_scheduler.StepLR(legacy, 1), 1)
    optimizer = build_optimizer(model, _CONFIG)
    with caplog.at_level(logging.INFO, logger="lemtrain.core_loop"):
        restore_optimizer_state(_train_state(model, optimizer), legacy.state_dict())
    assert "Migrated legacy" in caplog.text
    assert all("exp_avg" in optimizer.state[p] for p in model.parameters())


def test_resume_keeps_fresh_state_on_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    """A state that cannot be mapped leaves the fresh optimizer and logs a warning."""
    model = _Net()
    optimizer = build_optimizer(model, _CONFIG)
    other = _legacy_optimizer(nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 1)))
    with caplog.at_level(logging.WARNING, logger="lemtrain.core_loop"):
        restore_optimizer_state(_train_state(model, optimizer), other.state_dict())
    assert "keeping fresh optimizer state" in caplog.text
    assert not optimizer.state
