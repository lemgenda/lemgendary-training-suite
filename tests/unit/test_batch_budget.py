"""Resolution-jump batch refit: exact effective batch and an untouched model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests.fixtures.training_context import BUDGET_MODEL_INFO, fresh_governance
from training.governance.curriculum import CurriculumState
from training.hardware.discovery import discover_device
from training.hardware.policy import apply_hardware_policy
from training.hardware.sentinel import SentinelGuard
from training.training import (
    DataComponents,
    HardwareComponents,
    ModelStack,
    RunSettings,
    TrainingContext,
    TrainingPaths,
)
from training.training.batch_budget import refit_batch_for_resolution

_MODEL_INFO = BUDGET_MODEL_INFO


class _BnNet(nn.Module):
    """A BatchNorm network whose running statistics a train-mode pass would move."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x``."""
        return self.bn(x)


def _ctx(root: Path, model: nn.Module) -> TrainingContext:
    cpu = discover_device(force_device="cpu")
    data = TensorDataset(torch.randn(8, 3), torch.randn(8, 3))
    paths = TrainingPaths(
        project_root=root,
        local_checkpoint_dir=root / "ckpt",
        export_dir=root / "export",
        progress_local_path=root / "progress.pth",
        best_checkpoint_path=root / "best.pth",
        history_csv_path=root / "history.csv",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return TrainingContext(
        model_name="budget_model",
        models=ModelStack(
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            criterion=nn.MSELoss(),
        ),
        data=DataComponents(
            train_loader=DataLoader(data, batch_size=4), val_loader=DataLoader(data, batch_size=4)
        ),
        hardware=HardwareComponents(
            device_info=cpu,
            policy=apply_hardware_policy("budget_model", {}, cpu),
            sentinel=SentinelGuard(device=torch.device("cpu")),
        ),
        governance=fresh_governance(dict(_MODEL_INFO)),
        run=RunSettings(paths=paths, config={}, model_info=_MODEL_INFO, args=None, total_epochs=2),
    )


@pytest.mark.parametrize("effective", [1, 7, 12, 16, 24, 36, 64, 96])
@pytest.mark.parametrize("batch", [1, 4, 8, 12, 32])
@pytest.mark.parametrize("safe", [1, 3, 5, 8, 16])
def test_fit_keeps_effective_batch_exact(effective: int, batch: int, safe: int) -> None:
    """``b_micro * accum`` is unchanged and ``b_micro`` never exceeds the budget."""
    if effective % batch:
        return
    state = CurriculumState(current_batch=batch, current_acc=effective // batch)
    changed = state.fit_batch_to_budget(safe)
    assert state.current_batch * state.current_acc == effective
    assert changed == (batch > safe)
    if changed:
        assert state.current_batch <= safe
        # The largest divisor of the effective batch within the budget.
        assert all(effective % d for d in range(state.current_batch + 1, safe + 1))
    else:
        assert state.current_batch == batch


def test_refit_on_jump_shrinks_batch_and_keeps_the_model(tmp_path: Path) -> None:
    """The probe sees the new resolution; BatchNorm stats and train mode survive it."""
    model = _BnNet()
    model.eval()
    ctx = _ctx(tmp_path, model)
    governor = ctx.governance.governor
    governor.curriculum.current_res = 384
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seen: list[int] = []

    def probe(probed: nn.Module, resolution: int) -> int:
        seen.append(resolution)
        probed.train()
        probed(torch.randn(16, 3) * 9.0 + 4.0)
        return 5

    assert refit_batch_for_resolution(ctx, probe)
    assert seen == [384]
    assert (governor.current_batch, governor.current_acc) == (4, 6)
    assert not model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_refit_without_measurement_keeps_batch(tmp_path: Path) -> None:
    """Off CUDA no VRAM measurement exists, so the batch is left alone."""
    ctx = _ctx(tmp_path, _BnNet())
    assert not refit_batch_for_resolution(ctx)
    assert (ctx.governance.governor.current_batch, ctx.governance.governor.current_acc) == (12, 2)


@pytest.mark.parametrize(
    ("batch", "accum", "expected"),
    [
        (16, 1, (8, 2)),  # even split: plain halving
        (12, 2, (6, 4)),
        (6, 4, (3, 8)),
        (3, 8, (1, 24)),
        (7, 1, (1, 7)),  # prime effective batch: no split between 1 and 7
        (10, 3, (5, 6)),
        (1, 24, (1, 24)),  # already minimal: unchanged
    ],
)
def test_oom_throttle_keeps_effective_batch_exact(
    batch: int, accum: int, expected: tuple[int, int]
) -> None:
    """The OOM throttle uses the refit/resume exact-divisor rule, not rounding.

    ``target_effective_batch`` is deliberately off (5): the old rounding against it
    drifted the effective batch, the exact split keeps ``batch * accum``.
    """
    state = CurriculumState(current_batch=batch, current_acc=accum, target_effective_batch=5)
    assert state.throttle_on_oom() == expected
    assert state.current_batch * state.current_acc == batch * accum


def test_oom_throttle_respects_min_batch() -> None:
    """``min_batch`` bounds the halved budget; the split stays exact."""
    state = CurriculumState(current_batch=16, current_acc=3)
    assert state.throttle_on_oom(min_batch=12) == (12, 4)


def test_governor_oom_keeps_loader_effective_batch(tmp_path: Path) -> None:
    """``handle_oom`` (mid-epoch OOM path) halves the micro-batch with an exact split."""
    ctx = _ctx(tmp_path, _BnNet())
    governor = ctx.governance.governor
    assert governor.handle_oom(current_batch=12) == (6, 4)
    assert governor.current_batch * governor.current_acc == 24
