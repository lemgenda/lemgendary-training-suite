"""Registry ``parameters_m`` equals the parameter count of the model the code builds.

The registry stated figures nobody had measured (``retinaface`` 27.2 M for a 0.94 M
network). Each model with ``parameters_m`` is instantiated on the CPU from the registry and
its parameters are counted; the registry value is that count in millions, 4 decimals.
``yolov8n`` is built by Ultralytics from its architecture file, as its trainer does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from models.factory import get_model

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "unified_models_v2.yaml"
_REGISTRY = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
_COUNTED_KEYS = sorted(
    key
    for key, entry in _REGISTRY.items()
    if not key.startswith("_") and isinstance(entry, dict) and "parameters_m" in entry
)
_DECIMALS = 4
_YOLO_ARCHITECTURE = "yolov8n.yaml"


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _built_model(model_key: str) -> torch.nn.Module:
    """The network the trainer builds for ``model_key``."""
    if _REGISTRY[model_key].get("class_name") == "YOLO":
        ultralytics = pytest.importorskip("ultralytics")
        return ultralytics.YOLO(_YOLO_ARCHITECTURE).model
    return get_model(model_key)


def test_every_trainable_model_states_its_parameters() -> None:
    """Every model except the LLM assistant (a fine-tuned external base) has a count."""
    without = {
        key
        for key, entry in _REGISTRY.items()
        if not key.startswith("_") and isinstance(entry, dict) and "parameters_m" not in entry
    }
    assert without == {"ai_helper_lemgend"}


@pytest.mark.parametrize("model_key", _COUNTED_KEYS)
def test_registry_parameters_match_the_code(model_key: str) -> None:
    """``parameters_m`` is the instantiated model's parameter count in millions."""
    counted = _parameter_count(_built_model(model_key))
    assert _REGISTRY[model_key]["parameters_m"] == round(counted / 1e6, _DECIMALS)
