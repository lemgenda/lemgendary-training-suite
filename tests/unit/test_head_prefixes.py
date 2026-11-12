"""Every registered architecture declares its head explicitly (R2-T3)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from torch import nn

from models.factory import get_model
from models.generic_causal_llm import GenericCausalLLM
from training.training.lr_scaling import GROUP_ROLE_KEY, HEAD_ROLE
from training.training.param_groups import HEAD_PREFIXES_ATTR, build_param_groups, head_prefixes

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
# Trained by the Ultralytics trainer with its own optimizer groups.
_OWN_OPTIMIZER = {"yolov8n"}
# Causal LLMs are covered by the mock-model tests below (no weight download).
_LLM_KEYS = {
    key
    for key, info in _REGISTRY.items()
    if isinstance(info, dict) and info.get("dataset_type") == "text_instruction"
}
_KEYS = sorted(
    key
    for key, info in _REGISTRY.items()
    if isinstance(info, dict)
    and not key.startswith("_")
    and key not in _OWN_OPTIMIZER | _LLM_KEYS
)


def _head_and_backbone(model: nn.Module) -> tuple[int, int]:
    head = backbone = 0
    for group in build_param_groups(model, 0.01):
        count = sum(p.numel() for p in group["params"])
        if group[GROUP_ROLE_KEY] == HEAD_ROLE:
            head += count
        else:
            backbone += count
    return head, backbone


@pytest.mark.parametrize("model_key", _KEYS)
def test_registered_model_declares_a_sensible_head(
    model_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Declared prefixes all match trainable parameters; head and backbone are non-empty."""
    model = get_model(model_key, _REGISTRY[model_key])
    with caplog.at_level(logging.WARNING, logger="lemtrain.training.param_groups"):
        prefixes = head_prefixes(model)
    assert hasattr(model, HEAD_PREFIXES_ATTR), f"{model_key} has no {HEAD_PREFIXES_ATTR}"
    assert "declares no" not in caplog.text
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    for prefix in prefixes:
        assert any(n.startswith(prefix) for n in names), f"{model_key}: {prefix} matches nothing"
    head, backbone = _head_and_backbone(model)
    assert head > 0, f"{model_key}: empty head group"
    assert backbone > 0, f"{model_key}: empty backbone group"


def test_full_fine_tune_llm_head_is_lm_head() -> None:
    """A fully trainable causal LM puts its vocabulary projection in the head group."""
    model = GenericCausalLLM("mock")
    assert head_prefixes(model) == ("lm_head.",)
    head, backbone = _head_and_backbone(model)
    assert head == model.lm_head.weight.numel()
    assert backbone > 0


def test_frozen_lm_head_has_no_separate_head() -> None:
    """With the base model frozen (LoRA), no head group is split off."""
    model = GenericCausalLLM("mock")
    model.lm_head.weight.requires_grad_(False)
    assert not head_prefixes(model)


def test_unknown_model_falls_back_with_a_log(caplog: pytest.LogCaptureFixture) -> None:
    """A model without the attribute uses its last child, and says so."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    with caplog.at_level(logging.WARNING, logger="lemtrain.training.param_groups"):
        assert head_prefixes(model) == ("1.",)
    assert "declares no head_param_prefixes" in caplog.text
