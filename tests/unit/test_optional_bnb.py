"""The optional bitsandbytes 8-bit optimizer: used when present, AdamW when absent."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn, optim

from training.training import optimizer as optimizer_module
from training.training.optimizer import build_optimizer, paged_adamw_8bit


def _missing(name: str) -> ModuleType:
    raise ImportError(f"No module named {name!r}")


def test_absent_bitsandbytes_yields_no_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without bitsandbytes the adapter reports None instead of raising."""
    monkeypatch.setattr(importlib, "import_module", _missing)
    assert paged_adamw_8bit() is None


def test_absent_bitsandbytes_falls_back_to_adamw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested 8-bit optimizer degrades to ``torch.optim.AdamW``."""
    monkeypatch.setattr(optimizer_module, "paged_adamw_8bit", lambda: None)
    built = build_optimizer(nn.Linear(2, 2), {"optimizer": {"type": "paged_adamw_8bit"}})
    assert isinstance(built, optim.AdamW)


def test_present_bitsandbytes_factory_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the adapter finds PagedAdamW8bit, the requested optimizer is built from it."""
    calls: list[dict[str, Any]] = []

    def factory(param_groups: Any, **kwargs: Any) -> optim.Optimizer:
        calls.append(kwargs)
        return torch.optim.SGD(param_groups, lr=kwargs["lr"])

    monkeypatch.setattr(optimizer_module, "paged_adamw_8bit", lambda: factory)
    built = build_optimizer(nn.Linear(2, 2), {"optimizer": {"type": "adamw_8bit", "lr": 0.01}})
    assert isinstance(built, optim.SGD)
    assert calls and calls[0]["lr"] == pytest.approx(0.01)


def test_installed_bitsandbytes_class_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter hands back ``bitsandbytes.optim.PagedAdamW8bit`` when it is an optimizer."""
    fake_optim = ModuleType("bitsandbytes.optim")
    monkeypatch.setattr(fake_optim, "PagedAdamW8bit", optim.AdamW, raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_optim)
    assert paged_adamw_8bit() is optim.AdamW
