"""Unit tests for model export subsystem: ONNX, TorchScript, WebGPU, and export_all."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.export.common import (
    extract_input_dimensions,
    resolve_export_paths,
)
from training.export.manifest import (
    MANIFEST_FILENAME,
    build_export_manifest,
    compute_file_sha256,
    resolve_family_tolerances,
)
from training.export.onnx import export_onnx
from training.export.torch_standalone import export_torch_standalone
from training.export.webgpu import export_webgpu_onnx
from training.utils.lazy_imports import load_onnxruntime


class DummyExportNet(nn.Module):
    """Small deterministic network for export testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv, ReLU and the output conv."""
        return self.out(self.relu(self.conv(x)))


class TestExportCommon(unittest.TestCase):
    """Tests for export path resolution and dimension extraction."""

    def test_extract_input_dimensions(self) -> None:
        """Verify input dimensions are extracted from scalar and list input_size values."""
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
        """Verify export paths resolve to the canonical stem and production dir."""
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
        """Verify ONNX export writes both FP32 and FP16 graphs."""
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
        """Verify the standalone PyTorch export succeeds and writes the file."""
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
        """Verify the WebGPU ONNX export succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            out_webgpu = Path(tmp) / "model_webgpu.onnx"
            ok = export_webgpu_onnx(self.model, out_webgpu, dummy_input_shape=(1, 3, 64, 64))
            self.assertTrue(ok)
            self.assertTrue(out_webgpu.exists())
            self.assertGreater(out_webgpu.stat().st_size, 100)

    def test_export_manifest_generation_and_validation(self) -> None:
        """Verify export_manifest.json generation, hashing, and tolerance metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pt_file = tmp_path / "DummyNet.pt"
            onnx_fp32 = tmp_path / "DummyNet_FP32.onnx"
            onnx_fp16 = tmp_path / "DummyNet.onnx"

            export_torch_standalone(self.model, pt_file)
            export_onnx(self.model, onnx_fp32, dummy_shape=(1, 3, 64, 64), half=False)
            export_onnx(self.model, onnx_fp16, dummy_shape=(1, 3, 64, 64), half=True)

            files = {
                "pt": pt_file,
                "onnx_fp32": onnx_fp32,
                "onnx_fp16": onnx_fp16,
            }
            manifest = build_export_manifest("dummynet", files)
            saved_manifest = manifest.save(tmp_path)

            self.assertTrue(saved_manifest.is_file())
            self.assertEqual(saved_manifest.name, MANIFEST_FILENAME)

            content = json.loads(saved_manifest.read_text(encoding="utf-8"))
            self.assertEqual(content["model_key"], "dummynet")
            self.assertEqual(content["stem"], "dummynet")
            self.assertEqual(content["opset"], 18)
            self.assertIn("artifacts", content)
            self.assertIn("pt", content["artifacts"])
            self.assertIn("onnx_fp32", content["artifacts"])
            self.assertIn("onnx_fp16", content["artifacts"])

            # Verify checksums match reality
            for artifact in content["artifacts"].values():
                real_hash = compute_file_sha256(artifact["file_path"])
                self.assertEqual(artifact["sha256"], real_hash)

    def test_numerical_equivalence_fp32_onnx_vs_pytorch(self) -> None:
        """Verify ONNX Runtime inference matches PyTorch output within documented tolerance."""
        with tempfile.TemporaryDirectory() as tmp:
            out_onnx = Path(tmp) / "model_fp32.onnx"
            export_onnx(self.model, out_onnx, dummy_shape=(1, 3, 64, 64), half=False)

            # Load into ONNX Runtime
            ort = load_onnxruntime()
            session = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])

            # Fixed deterministic test batch
            torch.manual_seed(42)
            test_x = torch.randn(2, 3, 64, 64, dtype=torch.float32)

            with torch.no_grad():
                expected_pt = self.model(test_x).numpy()

            in_name = session.get_inputs()[0].name
            actual_onnx = session.run(None, {in_name: test_x.numpy()})[0]

            tols = resolve_family_tolerances("vision")["fp32"]
            max_abs_diff = float(np.max(np.abs(expected_pt - actual_onnx)))

            self.assertLessEqual(max_abs_diff, tols["atol"] * 5)
            np.testing.assert_allclose(actual_onnx, expected_pt, atol=1e-3, rtol=1e-2)

    def test_family_parity_tolerances(self) -> None:
        """Verify family-specific parity tolerances for vision, yolo, forex, and llm."""
        vision_tol = resolve_family_tolerances("nima_aesthetic_pro")
        self.assertEqual(vision_tol["fp32"]["atol"], 1e-3)

        yolo_tol = resolve_family_tolerances("yolov8n")
        self.assertEqual(yolo_tol["fp32"]["atol"], 1e-3)

        forex_tol = resolve_family_tolerances("forex_eurusd_h1")
        self.assertEqual(forex_tol["fp32"]["atol"], 1e-4)

        llm_tol = resolve_family_tolerances("ai_helper_lemgend")
        self.assertEqual(llm_tol["fp32"]["atol"], 1e-3)

    def test_verify_canary_rejects_non_finite_fp16(self):
        from training.export.base import ExportError
        from training.export.common import OnnxGraphSpec
        from training.export.onnx_variants import ONNX_FP16_VARIANT, OnnxVariantWriter

        class NanModule(nn.Module):
            def forward(self, x):
                return torch.full_like(x, float("nan"))

        mod = NanModule().eval()
        dummy = torch.randn(1, 3, 32, 32)
        writer = OnnxVariantWriter(mod, (dummy,), OnnxGraphSpec())

        class MockProgram:
            def __call__(self, *args):
                return (torch.tensor([float("nan")]),)

        with self.assertRaises(ExportError) as ctx:
            writer.verify_canary(MockProgram(), ONNX_FP16_VARIANT)
        self.assertEqual(ctx.exception.code, "ONNX_CANARY_FAILED")


if __name__ == "__main__":
    unittest.main()
