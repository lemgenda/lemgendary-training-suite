"""Exported ``.pt`` files carry what rebuilds them (R3-K3 item 6).

The forex ``.pt`` metadata holds ``model_kwargs``, so a non-default architecture loads
through ``load_forex_checkpoint``; the LLM ``.pt`` is the suite's dict format.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from models.forex_predictor import ForexPredictor
from training.export.forex_exporter import _pt_metadata
from training.export.llm_exporter import llm_pt_payload
from training.export.mt5_signal import load_forex_checkpoint
from training.export.torch_standalone import export_torch_standalone
from training.utils.model_payload import load_model_payload


def test_forex_pt_rebuilds_a_non_default_architecture(tmp_path: Path) -> None:
    """``model_kwargs`` in the ``.pt`` metadata rebuild the exact architecture and weights."""
    torch.manual_seed(0)
    model = ForexPredictor(
        [1, 5], d_model=32, n_heads=2, n_layers=1, head_hidden=48, pairs=["EURUSD", "GBPUSD"]
    )
    pt_path = tmp_path / "Forex.pt"
    export_torch_standalone(model, pt_path, metadata=_pt_metadata("forex_predictor", model))

    metadata = load_model_payload(pt_path, trust_pickle=True).metadata
    assert metadata["model_kwargs"]["d_model"] == 32
    assert metadata["model_kwargs"]["active_timeframes"] == [1, 5]

    rebuilt = load_forex_checkpoint(pt_path)
    assert rebuilt.active_timeframes == [1, 5]
    assert rebuilt.arch == model.arch
    original = model.state_dict()
    for name, tensor in rebuilt.state_dict().items():
        assert torch.equal(tensor, original[name]), name


def test_non_forex_pt_metadata_has_no_kwargs() -> None:
    """Only a ForexPredictor contributes ``model_kwargs``."""
    assert _pt_metadata("other", nn.Linear(2, 2)) == {"model_key": "other"}


def test_llm_pt_is_the_dict_format(tmp_path: Path) -> None:
    """The LLM ``.pt`` holds ``model_state`` and ``metadata`` and loads with weights_only."""
    model = nn.Linear(4, 3)
    pt_path = tmp_path / "Llm.pt"
    torch.save(llm_pt_payload(model, "ai_helper_lemgend"), pt_path)

    payload = load_model_payload(pt_path)
    assert payload.metadata == {"model_key": "ai_helper_lemgend", "architecture": "Linear"}
    assert set(payload.state_dict) == {"weight", "bias"}
    assert torch.equal(payload.state_dict["weight"], model.weight)
