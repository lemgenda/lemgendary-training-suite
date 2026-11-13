"""NIMA logit bound from the registry, as a soft cap that follows the governor (R2-T6)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import models.nima as nima_module
from models.factory import apply_registry_logit_cap
from models.nima import DEFAULT_LOGIT_CAP, NIMAModel, soft_cap_logits
from training.governance.logit_cap_sync import sync_logit_cap

_ROOT = Path(__file__).resolve().parents[2]


class _Capped(torch.nn.Module):
    """A model with the soft-cap interface only."""

    logit_cap = DEFAULT_LOGIT_CAP

    def set_logit_cap(self, cap: float) -> None:
        """Record the cap."""
        self.logit_cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity."""
        return x


def test_soft_cap_is_bounded_smooth_and_keeps_gradients() -> None:
    """Large logits stay inside the cap but still receive a gradient."""
    logits = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0], requires_grad=True)
    capped = soft_cap_logits(logits, 15.0)
    values = capped.detach()
    assert float(values.abs().max()) <= 15.0
    assert torch.allclose(values[1:4], logits.detach()[1:4], atol=1e-2)
    capped.sum().backward()
    assert logits.grad is not None
    assert bool((logits.grad >= 0).all())
    assert float(logits.grad[2]) == pytest.approx(1.0)


def test_cap_zero_turns_the_bound_off() -> None:
    """``0`` returns the logits unchanged."""
    logits = torch.tensor([50.0, -50.0])
    assert torch.equal(soft_cap_logits(logits, 0.0), logits)


def test_no_hardcoded_clamp_remains_in_nima() -> None:
    """The [-10, 10] clamp is gone from the model source."""
    source = inspect.getsource(nima_module)
    assert "torch.clamp(logits" not in source


def test_every_quality_model_takes_its_cap_from_the_registry() -> None:
    """Each NIMA registry entry declares ``logit_clamp`` and the factory applies it."""
    registry = yaml.safe_load((_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8"))
    nima_keys = [
        key
        for key, info in registry.items()
        if isinstance(info, dict) and info.get("class_name") == "NIMA_Model"
    ]
    assert nima_keys
    for key in nima_keys:
        cap = registry[key]["stabilizers"]["logit_clamp"]
        model = apply_registry_logit_cap(_Capped(), registry[key]["stabilizers"])
        assert model.logit_cap == cap, key


def test_nima_forward_uses_the_configured_cap() -> None:
    """The model output never exceeds the configured cap."""
    model = NIMAModel(backbone="mobilenet_v3_small").eval()
    model.set_logit_cap(0.5)
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert out.shape == (1, 10)
    assert float(out.abs().max()) <= 0.5


def test_governor_clamp_drives_the_model_cap() -> None:
    """The governor's clamp is applied once per change."""
    model = _Capped()
    governor = SimpleNamespace(current_clamp=20.0)
    assert sync_logit_cap(model, governor)
    assert model.logit_cap == 20.0
    assert not sync_logit_cap(model, governor)
    assert not sync_logit_cap(torch.nn.Linear(1, 1), governor)
