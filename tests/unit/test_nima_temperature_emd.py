"""NIMA temperature and EMD loss (R4-C4).

Temperature 1 in training (the governor no longer moves it, nor the soft cap 15, for quality
models), the paper EMD form as registry option ``stabilizers.emd_form`` (pinned per run and
recorded in checkpoints), the 1.0 loss fallback, no ``softmax_temp`` model parameter, and the
optional post-hoc temperature calibration.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from models.nima import NIMAModel
from training.checkpoint.resume import parse_resume_state
from training.context_resume import pin_resumed_emd_form
from training.export.common import wrap_quality_model
from training.governance.governor import SmartTrainingGovernor
from training.governance.logit_cap_sync import sync_logit_cap
from training.loss_emd import (
    LEGACY_EMD_FORM,
    NIMA_EMD_FORM,
    SQUARED_EMD_FORM,
    emd_loss,
    registry_emd_form,
)
from training.losses import CombinedLoss
from training.quality_calibration import calibrated_temperature, fit_temperature

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
_NIMA_KEYS = [k for k, v in _REGISTRY.items() if isinstance(v, dict) and v.get("loss_fn") == "emd"]


def _quality_info(key: str = "nima_technical") -> dict[str, Any]:
    return dict(_REGISTRY[key])


def test_registry_nima_entries_use_the_paper_form() -> None:
    """Every EMD-trained entry names its form explicitly (``nima``)."""
    assert len(_NIMA_KEYS) == 5
    for key in _NIMA_KEYS:
        assert registry_emd_form(_REGISTRY[key]["stabilizers"]) == NIMA_EMD_FORM


def test_registry_rejects_unknown_and_internal_forms() -> None:
    """``legacy_sum`` is for resumed pre-R4 runs only, never a registry choice."""
    assert registry_emd_form(None) == NIMA_EMD_FORM
    with pytest.raises(ValueError, match="emd_form"):
        registry_emd_form({"emd_form": LEGACY_EMD_FORM})
    with pytest.raises(ValueError, match="emd_form"):
        registry_emd_form({"emd_form": "cubic"})


def test_emd_forms_follow_their_definitions() -> None:
    """``nima`` = sqrt(mean + 1e-8), ``squared`` = mean, ``legacy_sum`` = sum (N x mean)."""
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    t = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    gap2 = (torch.cumsum(p, -1) - torch.cumsum(t, -1)) ** 2
    assert emd_loss(p, t, SQUARED_EMD_FORM) == pytest.approx(gap2.mean().item())
    assert emd_loss(p, t, LEGACY_EMD_FORM) == pytest.approx(gap2.sum().item())
    assert emd_loss(p, t, NIMA_EMD_FORM) == pytest.approx((gap2.mean() + 1e-8).sqrt().item())
    with pytest.raises(ValueError, match="EMD form"):
        emd_loss(p, t, "cubic")


def test_nima_form_has_a_finite_gradient_at_a_perfect_match() -> None:
    """The epsilon keeps the square-root gradient finite when prediction equals target."""
    logits = torch.zeros(2, 10, requires_grad=True)
    target = torch.full((2, 10), 0.1)
    emd_loss(torch.softmax(logits, -1), target, NIMA_EMD_FORM).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_fallback_temperature_is_one() -> None:
    """Without a stabilizer block the loss no longer falls back to temperature 0.1."""
    loss = CombinedLoss(task_type="quality")
    assert loss.stab["softmax_temp"] == 1.0
    assert loss.emd_form == NIMA_EMD_FORM


def test_quality_loss_ignores_any_temperature() -> None:
    """The quality loss is a plain softmax: a registry temperature does not change it."""
    torch.manual_seed(0)
    pred, target = torch.randn(4, 10), torch.rand(4, 10)
    stab = dict(_quality_info()["stabilizers"])
    base = CombinedLoss(task_type="quality", stabilizers=stab)(pred, target)
    hot = CombinedLoss(task_type="quality", stabilizers={**stab, "softmax_temp": 0.1})
    assert hot(pred, target).item() == pytest.approx(base.item())


def test_governor_keeps_temperature_and_cap_fixed_for_quality() -> None:
    """Sharpening, recoil, resets and checkpoint restore leave T = 1 and the cap 15."""
    info = _quality_info()
    governor = SmartTrainingGovernor(model_info=info, stabilizers=info["stabilizers"])
    assert governor.current_temp == 1.0
    governor.recoil()
    governor.thermal.temperature = 0.5
    governor.runtime.calibration.current_clamp = 20.0
    governor.load_state({"softmax_temp": 0.7, "logit_clamp": 30.0})
    assert governor.current_temp == 1.0
    assert governor.current_clamp == 15.0
    model = SimpleNamespace(logit_cap=15.0, set_logit_cap=lambda cap: None)
    assert not sync_logit_cap(model, governor)


def test_governor_still_drives_temperature_elsewhere() -> None:
    """Other tasks keep the thermal logic (only quality models are pinned)."""
    info = dict(_REGISTRY["forex_predictor"])
    governor = SmartTrainingGovernor(model_info=info, stabilizers=info["stabilizers"])
    governor.thermal.temperature = 0.9
    assert governor.current_temp == 0.9


def test_nima_has_no_temperature_parameter() -> None:
    """``softmax_temp`` and ``set_temp`` are gone; old state dicts still load non-strictly."""
    assert not hasattr(NIMAModel, "set_temp")
    assert "self.softmax_temp" not in inspect.getsource(NIMAModel)


def test_resume_state_carries_the_recorded_form() -> None:
    """``parse_resume_state`` reads the checkpoint's ``emd_form`` (``None`` when absent)."""
    payload: dict[str, Any] = {"epoch": 2, "model_state": {}, "emd_form": "squared"}
    assert parse_resume_state(payload).governance.emd_form == SQUARED_EMD_FORM
    assert parse_resume_state({"epoch": 2, "model_state": {}}).governance.emd_form is None


def test_resume_pins_the_recorded_or_legacy_form() -> None:
    """Fresh runs use the registry form; a resume keeps the checkpoint's (legacy if none)."""
    loss = CombinedLoss(task_type="quality", stabilizers={"emd_form": "nima"})
    assert pin_resumed_emd_form(None, loss) == LEGACY_EMD_FORM
    assert loss.emd_form == LEGACY_EMD_FORM
    loss = CombinedLoss(task_type="quality", stabilizers={"emd_form": "nima"})
    assert pin_resumed_emd_form("squared", loss) == SQUARED_EMD_FORM
    other = CombinedLoss(task_type="restoration")
    assert pin_resumed_emd_form(None, other) == NIMA_EMD_FORM
    assert other.emd_form == NIMA_EMD_FORM


def test_export_uses_the_calibrated_temperature() -> None:
    """``SoftmaxWrapper`` takes ``calibration.temperature`` (default 1.0)."""
    info: dict[str, Any] = {"dataset_type": "quality"}
    assert wrap_quality_model(nn.Identity(), info).temperature == 1.0
    info["calibration"] = {"temperature": 1.3}
    assert wrap_quality_model(nn.Identity(), info).temperature == 1.3
    assert isinstance(wrap_quality_model(nn.Identity(), {"dataset_type": "x"}), nn.Identity)
    with pytest.raises(ValueError, match="positive"):
        calibrated_temperature({"calibration": {"temperature": 0.0}})


def _scores(scale: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    targets = rng.dirichlet(np.ones(10), size=64)
    logits = np.log(targets + 1e-6) * scale + rng.normal(0, 0.05, targets.shape)
    return logits, targets


def test_calibration_accepts_a_temperature_that_keeps_correlations() -> None:
    """Over-confident logits (x2) fit T near 2 with lower EMD; PLCC/SRCC hold."""
    result = fit_temperature(*_scores(2.0))
    assert result.accepted
    assert result.temperature > 1.5
    assert result.best.emd < result.baseline.emd
    assert result.best.plcc >= result.baseline.plcc
    assert result.best.srcc >= result.baseline.srcc


def test_calibration_keeps_one_when_nothing_improves() -> None:
    """A grid without a better EMD keeps T = 1."""
    result = fit_temperature(*_scores(1.0), grid=(1.0,))
    assert not result.accepted
    assert result.temperature == 1.0
