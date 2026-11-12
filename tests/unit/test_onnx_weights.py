"""Sidecar rules of the ONNX release variants and the dynamo graph capture helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx import external_data_helper
from torch import nn

from training.export import onnx_weights
from training.export.common import OnnxGraphSpec, default_graph_spec
from training.export.dynamo_onnx import build_verified_program, dynamic_shapes_for
from training.export.onnx_variants import (
    ONNX_FP16_VARIANT,
    ONNX_FP32_VARIANT,
    OnnxVariantWriter,
)
from training.export.onnx_weights import WeightLayout, relocate_onnx, save_proto, sidecar_path
from training.utils.lazy_imports import load_onnxruntime


class _ConvNet(nn.Module):
    """Small fully convolutional network."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.out = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv, ReLU, conv."""
        return self.out(torch.relu(self.conv(x)))


class _WidthBranch(nn.Module):
    """Takes a different branch above width 40, like Swin's window shift."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Doubles the output for wide inputs."""
        y = self.conv(x)
        if x.shape[-1] > 40:
            return y * 2.0
        return y


class _VarArgs(nn.Module):
    """Collects its inputs in ``*args`` (like the Forex wrapper)."""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Sum of the inputs."""
        return args[0] + args[1]


def _writer(model: nn.Module, size: int = 32) -> OnnxVariantWriter:
    sample = torch.randn(2, 3, size, size)
    return OnnxVariantWriter(model.eval(), (sample,), default_graph_spec(sample))


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    ort = load_onnxruntime()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def _embedded(path: Path) -> bool:
    model = onnx.load(str(path), load_external_data=False)
    return not any(external_data_helper.uses_external_data(t) for t in model.graph.initializer)


def test_fp32_always_has_sidecar_and_fp16_is_embedded(tmp_path: Path) -> None:
    """FP32 writes ``<file>.data``; FP16 below the limit embeds its weights."""
    files = _writer(_ConvNet()).write_all(tmp_path, "Model")
    assert files["onnx_fp32"].name == "Model_FP32.onnx"
    assert files["onnx_fp32_data"] == tmp_path / "Model_FP32.onnx.data"
    assert files["onnx_fp32_data"].stat().st_size > 0
    assert not _embedded(files["onnx_fp32"])
    assert files["onnx_fp16"].name == "Model.onnx"
    assert "onnx_fp16_data" not in files
    assert _embedded(files["onnx_fp16"])
    assert not sidecar_path(files["onnx_fp16"]).exists()


def test_fp16_above_protobuf_limit_uses_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model larger than the (lowered) limit stores FP16 weights like FP32, and still runs."""
    monkeypatch.setattr(onnx_weights, "PROTOBUF_LIMIT_BYTES", 64)
    model = _ConvNet().eval()
    files = _writer(model).write(ONNX_FP16_VARIANT, tmp_path / "Big.onnx")
    assert files["onnx_fp16_data"] == tmp_path / "Big.onnx.data"
    assert not _embedded(files["onnx_fp16"])
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = _run(files["onnx_fp16"], x.astype(np.float16)).astype(np.float32)
    assert float(np.abs(got - expected).max()) < 2e-2


def test_save_proto_removes_stale_sidecar(tmp_path: Path) -> None:
    """Re-saving an embedded model deletes the sidecar of a previous sidecar export."""
    path = tmp_path / "m.onnx"
    files = _writer(_ConvNet()).write(ONNX_FP32_VARIANT, path)
    assert sidecar_path(path).exists()
    save_proto(onnx.load(str(path)), path, WeightLayout.EMBEDDED)
    assert files["onnx_fp32"] == path
    assert not sidecar_path(path).exists()
    assert _embedded(path)


def test_relocate_rewrites_sidecar_location(tmp_path: Path) -> None:
    """A renamed graph reads its weights from the renamed sidecar."""
    model = _ConvNet().eval()
    src = tmp_path / "a" / "Old_FP32.onnx"
    _writer(model).write(ONNX_FP32_VARIANT, src)
    dest = tmp_path / "b" / "New_FP32.onnx"
    data = relocate_onnx(src, dest)
    assert data == tmp_path / "b" / "New_FP32.onnx.data"
    assert not src.exists() and not sidecar_path(src).exists()
    x = np.ones((1, 3, 32, 32), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(_run(dest, x), expected, atol=1e-5, rtol=1e-4)


def test_shape_branch_pins_spatial_axes() -> None:
    """A width-dependent branch fails the probe, so height and width are pinned."""
    sample = torch.randn(2, 3, 32, 32)
    verified = build_verified_program(_WidthBranch().eval(), (sample,), default_graph_spec(sample))
    assert verified.graph.dynamic_axes == {"input": {0: "batch"}}
    assert [p.ok for p in verified.probes] == [False, True]


def test_fully_convolutional_graph_keeps_dynamic_axes() -> None:
    """Without shape branches the image axes stay dynamic after one probe."""
    sample = torch.randn(2, 3, 32, 32)
    verified = build_verified_program(_ConvNet().eval(), (sample,), default_graph_spec(sample))
    assert verified.graph.dynamic_axes == {"input": {0: "batch", 2: "height", 3: "width"}}
    assert len(verified.probes) == 1 and verified.probes[0].ok


def test_dynamic_shapes_name_shared_axes_once() -> None:
    """Varargs inputs nest in one tuple; a repeated axis name becomes ``Dim.DYNAMIC``."""
    graph = OnnxGraphSpec(
        input_names=["a", "b"],
        output_names=["y"],
        dynamic_axes={"a": {0: "batch"}, "b": {0: "batch"}},
    )
    shapes: Any = dynamic_shapes_for(_VarArgs(), graph, 2)
    assert shapes == (({0: "batch"}, {0: torch.export.Dim.DYNAMIC}),)
    assert dynamic_shapes_for(_ConvNet(), OnnxGraphSpec(), 1) is None
