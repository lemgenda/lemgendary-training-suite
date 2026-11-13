"""One loader for release ``.pt`` payloads, checkpoints and bare state dicts (R2-T1).

The release ``.pt`` is ``{"model_state", "model", "metadata"}``; older releases (and the
LLM release) are bare state dicts. Every consumer reads both through
:mod:`training.utils.model_payload`, with ``weights_only=True`` first (the torch >= 2.6
default) and full unpickling only for trusted files. Generated code (notebook cells,
model cards, the Kaggle README) embeds the same policy.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from training.cloud.metadata_generator import KaggleMetadataService
from training.cloud.variations import PYTORCH_FP32
from training.export.common import load_checkpoint_weights
from training.export.soup import create_pareto_model_soup
from training.export.torch_standalone import export_torch_standalone
from training.model_card_snippets import SnippetInput, _quality_usage
from training.registry.preprocessing import IMAGENET
from training.services.checkpoint_service import CheckpointService
from training.utils.model_payload import (
    ModelPayloadError,
    UntrustedPickleError,
    load_model_payload,
    model_payload_of,
    pt_load_source,
)


def _net(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def _release(path: Path, model: nn.Module) -> Path:
    assert export_torch_standalone(model, path, metadata={"model_key": "tiny"})
    return path


def _bare(path: Path, model: nn.Module) -> Path:
    torch.save(model.state_dict(), path)
    return path


def _assert_state(state: dict[str, torch.Tensor], model: nn.Module) -> None:
    expected = model.state_dict()
    assert state.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(state[key], value)


def test_normalizes_every_payload_shape() -> None:
    """Release payload, checkpoint, bare state dict and module give the same weights."""
    model = _net(0)
    state = model.state_dict()
    release = model_payload_of({"model_state": state, "model": model, "metadata": {"a": 1}})
    assert (release.kind, release.module, release.metadata) == ("release", model, {"a": 1})
    checkpoint = model_payload_of({"model_state_dict": state, "epoch": 3})
    assert checkpoint.kind == "checkpoint"
    assert checkpoint.raw["epoch"] == 3
    assert model_payload_of(dict(state)).kind == "state_dict"
    assert model_payload_of(model).kind == "module"
    assert model_payload_of({"model": model}).kind == "module"
    for payload in (release, checkpoint, model_payload_of(dict(state)), model_payload_of(model)):
        _assert_state(payload.state_dict, model)
    with pytest.raises(ModelPayloadError):
        model_payload_of({"epoch": 1})


def test_bare_state_dict_loads_weights_only(tmp_path: Path) -> None:
    """An old bare state dict needs no trust: it loads under ``weights_only=True``."""
    model = _net(1)
    payload = load_model_payload(_bare(tmp_path / "old.pt", model))
    assert payload.kind == "state_dict"
    _assert_state(payload.state_dict, model)


def test_release_payload_needs_trust_for_its_pickled_module(tmp_path: Path) -> None:
    """The release ``model`` entry is pickled; it is only unpickled when trusted."""
    model = _net(2)
    path = _release(tmp_path / "Tiny.pt", model)
    with pytest.raises(UntrustedPickleError):
        load_model_payload(path)
    payload = load_model_payload(path, trust_pickle=True)
    assert payload.kind == "release"
    assert payload.metadata == {"model_key": "tiny"}
    assert isinstance(payload.module, nn.Module)
    _assert_state(payload.state_dict, model)


@pytest.mark.parametrize("writer", [_release, _bare])
def test_export_weights_load_from_both_formats(tmp_path: Path, writer: Any) -> None:
    """``load_checkpoint_weights`` restores the same weights from either format."""
    model = _net(3)
    payload = load_model_payload(writer(tmp_path / "Tiny.pt", model), trust_pickle=True)
    target = _net(4)
    load_checkpoint_weights(target, payload.state_dict)
    _assert_state(target.state_dict(), model)


def test_soup_blends_a_release_and_a_bare_donor(tmp_path: Path) -> None:
    """Donors in both formats blend; the stale pickled module is not carried over."""
    first, second = _net(5), _net(6)
    paths = [_release(tmp_path / "a.pt", first), _bare(tmp_path / "b.pt", second)]
    soup = create_pareto_model_soup(paths, weights=[0.5, 0.5])
    assert "model" not in soup
    for key, value in soup["model_state"].items():
        expected = (first.state_dict()[key] + second.state_dict()[key]) / 2
        torch.testing.assert_close(value, expected)


def test_inspect_counts_parameters_of_a_bare_state_dict(tmp_path: Path) -> None:
    """A bare ``.pt`` reports its real parameter count (it used to report 0)."""
    model = _net(7)
    info = CheckpointService(project_root=tmp_path).inspect_checkpoint(
        _bare(tmp_path / "old.pt", model)
    )
    assert info["parameter_count"] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("writer", [_release, _bare])
def test_generated_loader_reads_both_formats(tmp_path: Path, writer: Any) -> None:
    """The loader embedded in generated code yields the weights for either format."""
    model = _net(8)
    path = writer(tmp_path / "Tiny.pt", model)
    snippet = tmp_path / "snippet.py"
    snippet.write_text("".join(pt_load_source("model_path", "device")), encoding="utf-8")
    namespace = runpy.run_path(
        str(snippet), init_globals={"torch": torch, "model_path": str(path), "device": "cpu"}
    )
    _assert_state(namespace["state"], model)
    assert (namespace["pt_model"] is None) == (writer is _bare)


def test_generated_code_uses_the_shared_loader() -> None:
    """Model cards and the Kaggle README embed the shared loader, nothing else."""
    loader = "".join(pt_load_source("model_path", "device"))
    assert loader in _quality_usage("nima", SnippetInput(224, 224, preprocessing=IMAGENET))
    readme = KaggleMetadataService.variation_usage("owner", "slug", "Tiny", PYTORCH_FP32)
    assert "".join(pt_load_source("model_path", '"cpu"')) in readme
    assert "weights_only=True" in readme
