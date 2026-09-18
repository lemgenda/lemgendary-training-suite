"""Unit tests for model export subsystem: ONNX, TorchScript, WebGPU, and export_all."""

import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from training.export.common import (
    ExportError,
    extract_input_dimensions,
    resolve_checkpoint_file,
    resolve_export_paths,
)
from training.export.onnx import export_onnx
from training.export.torch_standalone import export_torch_standalone
from training.export.webgpu import export_webgpu_onnx


class DummyExportNet(nn.Module):
    """Small deterministic network for export testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.relu(self.conv(x)))


class TestExportCommon(unittest.TestCase):
    """Tests for export path resolution and dimension extraction."""

    def test_extract_input_dimensions(self) -> None:
        # Scalar
        h, w = extract_input_dimensions({"input_size": 224})
        self.assertEqual((h, w), (224, 224))

        # List/Tuple (H, W)
        h, w = extract_input_dimensions({"input_size": [384, 512]})
        self.assertEqual((h, w), (384, 512))

        # Capped at 512
        h, w = extract_input_dimensions({"input_size": 1024})
        self.assertEqual((h, w), (512, 512))

    def test_resolve_export_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_name, prod_dir = resolve_export_paths(
                model_key="nafnet_debluring",
                model_info={"filename": "NafNetDeblur"},
                config={"export_dir": "models_export"},
                project_root=root,
            )
            self.assertEqual(base_name, "LemGendaryNafNetDeblur")
            self.assertTrue(prod_dir.exists())
            self.assertIn("models_export", str(prod_dir))


class TestModelExporters(unittest.TestCase):
    """Tests for ONNX, standalone PyTorch, and WebGPU exporters."""

    def setUp(self) -> None:
        self.model = DummyExportNet()
        self.model.eval()

    def test_onnx_export_fp32_and_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_fp32 = Path(tmp) / "model_fp32.onnx"
            out_fp16 = Path(tmp) / "model_fp16.onnx"

            # FP32 Export
            ok32 = export_onnx(self.model, out_fp32, dummy_shape=(1, 3, 64, 64), half=False)
            self.assertTrue(ok32)
            self.assertTrue(out_fp32.exists())
            self.assertGreater(out_fp32.stat().st_size, 100)

            # FP16 Export
            ok16 = export_onnx(self.model, out_fp16, dummy_shape=(1, 3, 64, 64), half=True)
            self.assertTrue(ok16)
            self.assertTrue(out_fp16.exists())
            self.assertGreater(out_fp16.stat().st_size, 100)

    def test_torch_standalone_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_pt = Path(tmp) / "model_standalone.pt"
            ok = export_torch_standalone(self.model, out_pt, metadata={"version": "1.0"})
            self.assertTrue(ok)
            self.assertTrue(out_pt.exists())

            # Verify reload
            loaded = torch.load(str(out_pt), map_location="cpu", weights_only=False)
            self.assertIn("model_state", loaded)
            self.assertIn("model", loaded)
            self.assertEqual(loaded.get("metadata", {}).get("version"), "1.0")

    def test_webgpu_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_webgpu = Path(tmp) / "model_webgpu.onnx"
            ok = export_webgpu_onnx(self.model, out_webgpu, dummy_input_shape=(1, 3, 64, 64))
            self.assertTrue(ok)
            self.assertTrue(out_webgpu.exists())
            self.assertGreater(out_webgpu.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
