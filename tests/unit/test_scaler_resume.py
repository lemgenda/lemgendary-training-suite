"""The AMP ``GradScaler`` loss scale survives a checkpoint resume (audit 3.3.2)."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from tests.unit.test_swa import _build_ctx
from training.context_data import RunSpec
from training.context_resume import TrainState, _restore_weights_and_optimizer
from training.training.step_progress import progress_payload

SAVED_SCALE = 1024.0


def _scaler(scale: float) -> torch.amp.GradScaler:
    """A CPU ``GradScaler`` (enabled, so its state is non-empty) at ``scale``."""
    return torch.amp.GradScaler("cpu", init_scale=scale)


def test_progress_payload_saves_and_resume_restores_the_scaler() -> None:
    """``progress_payload`` stores the scaler state; the resume loads it into the new scaler."""
    root = Path(tempfile.mkdtemp())
    try:
        ctx = _build_ctx(root, {})
        ctx.models.scaler = _scaler(SAVED_SCALE)
        payload = progress_payload(ctx, 1, ctx.models.model, {})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert payload["scaler_state_dict"]["scale"] == SAVED_SCALE

    restored = _scaler(65536.0)
    model = torch.nn.Linear(8, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train = TrainState(model, model, optimizer, None, None, None, scaler=restored)
    spec = RunSpec(
        args=argparse.Namespace(reset_scheduler=False),
        config={},
        model_info={},
        model_key="tiny",
        project_root=root,
        task_type="restoration",
        device=torch.device("cpu"),
    )
    _restore_weights_and_optimizer(spec, {"scaler_state_dict": payload["scaler_state_dict"]}, train)
    assert restored.get_scale() == SAVED_SCALE


def test_checkpoint_without_scaler_keeps_the_fresh_scale() -> None:
    """Older checkpoints without scaler state leave the new scaler untouched."""
    restored = _scaler(65536.0)
    model = torch.nn.Linear(8, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train = TrainState(model, model, optimizer, None, None, None, scaler=restored)
    spec = RunSpec(
        args=argparse.Namespace(reset_scheduler=False),
        config={},
        model_info={},
        model_key="tiny",
        project_root=Path("."),
        task_type="restoration",
        device=torch.device("cpu"),
    )
    _restore_weights_and_optimizer(spec, {}, train)
    assert restored.get_scale() == 65536.0
