"""Unit tests for StrictIsolationValidator and 3-variation isolation contract."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from training.export.validation import StrictIsolationValidator


class TestModelIsolation(unittest.TestCase):
    """Verifies that variation folders strictly isolate binary formats with zero cross-contamination."""

    def setUp(self) -> None:
        self.validator = StrictIsolationValidator()

    def test_pytorch_variation_valid_and_invalid(self) -> None:
        """Verify pytorch variation accepts strictly single *.pt and rejects foreign files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_dir = Path(tmp_dir) / "pytorch"
            pt_dir.mkdir(parents=True)

            # 1. Empty directory should fail
            rep1 = self.validator.validate_pytorch_variation(pt_dir, expected_stem="TestModel")
            self.assertFalse(rep1.is_valid)
            self.assertTrue(any("No .pt file found" in e for e in rep1.errors))

            # 2. Valid single *.pt file
            pt_file = pt_dir / "TestModel.pt"
            pt_file.write_bytes(b"dummy_pt_content")
            rep2 = self.validator.validate_pytorch_variation(pt_dir, expected_stem="TestModel")
            self.assertTrue(rep2.is_valid)
            self.assertEqual(len(rep2.errors), 0)

            # 3. Foreign file contamination (e.g. .pth or .txt)
            foreign_file = pt_dir / "TestModel_best.pth"
            foreign_file.write_bytes(b"foreign")
            rep3 = self.validator.validate_pytorch_variation(pt_dir, expected_stem="TestModel")
            self.assertFalse(rep3.is_valid)
            self.assertTrue(any("Foreign file detected" in e for e in rep3.errors))

    def test_onnx_fp32_variation_valid_and_invalid(self) -> None:
        """Verify onnx_fp32 variation accepts *_FP32.onnx and optional *.data sidecar only."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp32_dir = Path(tmp_dir) / "onnx_fp32"
            fp32_dir.mkdir(parents=True)

            # 1. Valid onnx FP32 file with sidecar
            onnx_file = fp32_dir / "TestModel_FP32.onnx"
            onnx_file.write_bytes(b"dummy_onnx_content")
            sidecar = fp32_dir / "TestModel_FP32.onnx.data"
            sidecar.write_bytes(b"dummy_weights")

            rep1 = self.validator.validate_onnx_fp32_variation(fp32_dir, expected_stem="TestModel")
            self.assertTrue(rep1.is_valid)

            # 2. Contamination with FP16 or foreign file
            rogue_file = fp32_dir / "TestModel.pt"
            rogue_file.write_bytes(b"rogue")
            rep2 = self.validator.validate_onnx_fp32_variation(fp32_dir, expected_stem="TestModel")
            self.assertFalse(rep2.is_valid)
            self.assertTrue(any("Foreign file detected" in e for e in rep2.errors))

    def test_onnx_fp16_variation_valid_and_invalid(self) -> None:
        """Verify onnx_fp16 variation accepts single self-contained *.onnx file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fp16_dir = Path(tmp_dir) / "onnx_fp16"
            fp16_dir.mkdir(parents=True)

            # 1. Valid embedded onnx file
            onnx_file = fp16_dir / "TestModel.onnx"
            onnx_file.write_bytes(b"dummy_fp16_content")

            rep1 = self.validator.validate_onnx_fp16_variation(fp16_dir, expected_stem="TestModel")
            self.assertTrue(rep1.is_valid)

            # 2. External sidecar in fp16 variation is rejected (FP16 must be embedded)
            sidecar = fp16_dir / "TestModel.onnx.data"
            sidecar.write_bytes(b"sidecar")
            rep2 = self.validator.validate_onnx_fp16_variation(fp16_dir, expected_stem="TestModel")
            self.assertFalse(rep2.is_valid)

    def test_validate_all_variations_aggregate_report(self) -> None:
        """Verify aggregate validation across dist/kaggle/ directory tree."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_kaggle = Path(tmp_dir) / "dist" / "kaggle"
            (dist_kaggle / "pytorch_default" / "checkpoints").mkdir(parents=True)
            (dist_kaggle / "pytorch").mkdir(parents=True)
            (dist_kaggle / "onnx_fp32").mkdir(parents=True)
            (dist_kaggle / "onnx_fp16").mkdir(parents=True)

            (dist_kaggle / "pytorch_default" / "checkpoints" / "best.pth").write_bytes(b"pth")
            (dist_kaggle / "pytorch" / "ModelA.pt").write_bytes(b"pt")
            (dist_kaggle / "onnx_fp32" / "ModelA_FP32.onnx").write_bytes(b"onnx32")
            (dist_kaggle / "onnx_fp16" / "ModelA.onnx").write_bytes(b"onnx16")

            report = self.validator.validate_all("ModelA", base_dir=tmp_dir)
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.variations), 4)


if __name__ == "__main__":
    unittest.main()
