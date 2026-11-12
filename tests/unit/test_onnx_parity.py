"""ONNX release-variant parity: every CPU-instantiable registry model, exported and run in ORT.

Each model is built with random weights (pretrained torchvision downloads are bypassed so the
test is offline), exported through the production exporter into the three release variants
and compared with PyTorch in ONNX Runtime on the CPU:

* ``<Stem>_FP32.onnx`` (+ ``.onnx.data``) against FP32 PyTorch: atol 1e-4, rtol 1e-3.
* ``<Stem>.onnx`` (FP16, embedded) against FP32 PyTorch: 2 % of the output scale.
* ``<Stem>.pt`` reloaded: identical outputs.

Graphs that keep dynamic height/width are also checked at a second resolution and batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import onnx
import pytest
import torch
from onnx import external_data_helper
from torch import nn

from models.factory import get_model
from training.export.base import BaseModelExporter
from training.export.dynamo_onnx import flatten_outputs, ort_run_errors
from training.export.exporter_factory import ExporterFactory
from training.export.forex_exporter import ForexModelExporter
from training.export.mt5_signal import forex_onnx_graph
from training.export.pytorch_exporter import PyTorchModelExporter
from training.export.validation import resolve_model_stem
from training.registry import get_model_registry
from training.utils.lazy_imports import load_onnxruntime

SAMPLE_SIZE = 64
FP32_ATOL, FP32_RTOL = 1e-4, 1e-3
FP16_SCALE_TOL = 2e-2


def _image_model_keys() -> list[str]:
    """Registry models the generic PyTorch exporter handles (YOLO, LLM and Forex excluded)."""
    keys = []
    for spec in get_model_registry().list_models():
        exporter: BaseModelExporter = ExporterFactory.get_exporter(spec.key, spec)
        if isinstance(exporter, PyTorchModelExporter):
            keys.append(spec.key)
    return keys


def _build_random(model_key: str) -> nn.Module:
    """Registry model with random weights: torchvision's pretrained state is not loaded."""

    def keep_random_init(_module: nn.Module, *_args: Any, **_kwargs: Any) -> None:
        return None

    with mock.patch.object(nn.Module, "load_state_dict", keep_random_init):
        model = get_model(model_key)
    return model.eval()


def _session(path: Path) -> Any:
    ort = load_onnxruntime()
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(session: Any, inputs: tuple[torch.Tensor, ...], dtype: np.dtype) -> list[np.ndarray]:
    feed = {}
    for meta, tensor in zip(session.get_inputs(), inputs, strict=True):
        array = tensor.numpy()
        feed[meta.name] = array.astype(dtype) if tensor.is_floating_point() else array
    return [np.asarray(out, dtype=np.float32) for out in session.run(None, feed)]


def _reference(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> list[np.ndarray]:
    with torch.no_grad():
        return [t.float().numpy() for t in flatten_outputs(model(*inputs))]


def _max_diff(expected: list[np.ndarray], actual: list[np.ndarray]) -> float:
    assert len(expected) == len(actual)
    worst = 0.0
    for want, got in zip(expected, actual, strict=True):
        assert want.shape == got.shape
        worst = max(worst, float(np.abs(want - got).max()) if want.size else 0.0)
    return worst


def _assert_fp32(expected: list[np.ndarray], actual: list[np.ndarray]) -> float:
    for want, got in zip(expected, actual, strict=True):
        np.testing.assert_allclose(got, want, atol=FP32_ATOL, rtol=FP32_RTOL)
    return _max_diff(expected, actual)


def _assert_fp16(expected: list[np.ndarray], actual: list[np.ndarray]) -> float:
    for want, got in zip(expected, actual, strict=True):
        scale = max(1.0, float(np.abs(want).max()) if want.size else 1.0)
        assert float(np.abs(want - got).max()) <= FP16_SCALE_TOL * scale
    return _max_diff(expected, actual)


def _has_dynamic_spatial(path: Path) -> bool:
    """Whether the graph's first input keeps symbolic height and width."""
    dims = onnx.load(str(path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim
    return len(dims) == 4 and all(d.dim_param for d in dims[2:])


def _assert_release_layout(files: dict[str, Path], stem: str) -> None:
    """Three variants under their release names; FP32 sidecar, embedded FP16."""
    assert files["pt"].name == f"{stem}.pt"
    assert files["onnx_fp32"].name == f"{stem}_FP32.onnx"
    assert files["onnx_fp32_data"].name == f"{stem}_FP32.onnx.data"
    assert files["onnx_fp32_data"].stat().st_size > 0
    assert files["onnx_fp16"].name == f"{stem}.onnx"
    assert "onnx_fp16_data" not in files
    assert not (files["onnx_fp16"].parent / f"{stem}.onnx.data").exists()
    fp16 = onnx.load(str(files["onnx_fp16"]), load_external_data=False)
    assert not any(external_data_helper.uses_external_data(t) for t in fp16.graph.initializer)


@pytest.fixture(name="ort_errors")
def fixture_ort_errors() -> Iterator[tuple[type[BaseException], ...]]:
    """ONNX Runtime's run-time exception types."""
    yield ort_run_errors()


@pytest.mark.parametrize("model_key", _image_model_keys())
def test_image_model_release_variants_match_pytorch(model_key: str, tmp_path: Path) -> None:
    """FP32 / FP16 ONNX and the .pt export reproduce the PyTorch model."""
    torch.manual_seed(0)
    model = _build_random(model_key)
    sample = torch.randn(2, 3, SAMPLE_SIZE, SAMPLE_SIZE)
    files = PyTorchModelExporter().export(model, model_key, output_dir=tmp_path, dummy_input=sample)
    stem = resolve_model_stem(model_key)
    _assert_release_layout(files, stem)

    expected = _reference(model, (sample,))
    fp32 = _session(files["onnx_fp32"])
    fp32_diff = _assert_fp32(expected, _run(fp32, (sample,), np.dtype(np.float32)))
    fp16_diff = _assert_fp16(
        expected, _run(_session(files["onnx_fp16"]), (sample,), np.dtype(np.float16))
    )

    if _has_dynamic_spatial(files["onnx_fp32"]):
        other = torch.randn(3, 3, SAMPLE_SIZE + 32, SAMPLE_SIZE + 32)
        _assert_fp32(_reference(model, (other,)), _run(fp32, (other,), np.dtype(np.float32)))

    payload = torch.load(str(files["pt"]), map_location="cpu", weights_only=False)
    reloaded: nn.Module = payload["model"].eval()
    np.testing.assert_array_equal(_reference(reloaded, (sample,))[0], expected[0])
    print(f"{model_key}: fp32 max abs diff {fp32_diff:.3e}, fp16 max abs diff {fp16_diff:.3e}")


def test_swin_backbone_graph_pins_resolution(tmp_path: Path, ort_errors: Any) -> None:
    """Swin's window shift depends on the feature size: the graph keeps a fixed resolution."""
    model = _build_random("nima_aesthetic_pro")
    sample = torch.randn(2, 3, SAMPLE_SIZE, SAMPLE_SIZE)
    files = PyTorchModelExporter().export(
        model, "nima_aesthetic_pro", output_dir=tmp_path, dummy_input=sample
    )
    assert not _has_dynamic_spatial(files["onnx_fp32"])
    session = _session(files["onnx_fp32"])
    batch3 = torch.randn(3, 3, SAMPLE_SIZE, SAMPLE_SIZE)
    _assert_fp32(_reference(model, (batch3,)), _run(session, (batch3,), np.dtype(np.float32)))
    with pytest.raises(ort_errors):
        _run(session, (torch.randn(1, 3, 96, 96),), np.dtype(np.float32))


def test_forex_release_variants_match_pytorch(tmp_path: Path) -> None:
    """Forex FP32 / FP16 (FP32 I/O for MT5) graphs reproduce the model at another batch."""
    torch.manual_seed(0)
    model = _build_random("forex_predictor")
    files = ForexModelExporter().export(model, "forex_predictor", output_dir=tmp_path)
    _assert_release_layout(files, resolve_model_stem("forex_predictor"))

    fx = forex_onnx_graph(model)
    batch3 = tuple(
        torch.randn(3, *t.shape[1:]) if t.is_floating_point() else torch.tensor([0, 2, 1])
        for t in fx.inputs
    )
    expected = _reference(fx.module, batch3)
    fp32_diff = _assert_fp32(
        expected, _run(_session(files["onnx_fp32"]), batch3, np.dtype(np.float32))
    )
    fp16_diff = _assert_fp16(
        expected, _run(_session(files["onnx_fp16"]), batch3, np.dtype(np.float32))
    )
    print(f"forex_predictor: fp32 max abs diff {fp32_diff:.3e}, fp16 max abs diff {fp16_diff:.3e}")
