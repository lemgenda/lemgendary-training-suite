"""Unit tests for Model Family Verification Engine and Contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from tests.fixtures.models import SingleConvModel
from training.checkpoint.manager import safe_atomic_save
from training.telemetry.schemas import TaskFamily
from training.verification.contracts import (
    VerificationStage,
)
from training.verification.engine import ModelVerificationEngine
from training.verification.registry import get_verification_contract, resolve_model_family


class NonDeterministicModel(nn.Module):
    """Model that violates forward determinism by adding random noise."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Random noise injected without seeding
        return out + torch.randn_like(out) * 0.5


class TestModelVerification(unittest.TestCase):
    """Tests for model verification contracts and engine stages."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.engine = ModelVerificationEngine(project_root=self.temp_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolve_model_family(self) -> None:
        """Verify model key to TaskFamily resolution."""
        self.assertEqual(resolve_model_family("forex_eurusd"), TaskFamily.FOREX)
        self.assertEqual(resolve_model_family("yolov8n"), TaskFamily.YOLO)
        self.assertEqual(resolve_model_family("retinaface_resnet50"), TaskFamily.FACE_DETECTION)
        self.assertEqual(resolve_model_family("ai_helper_lemgend"), TaskFamily.LLM)
        self.assertEqual(resolve_model_family("nafnet"), TaskFamily.VISION)

    def test_loadability_with_valid_checkpoint(self) -> None:
        """Test loadability verification on valid model state dict."""
        model = SingleConvModel()
        ckpt_file = self.temp_path / "valid_model.pth"
        payload = {
            "model_state_dict": model.state_dict(),
            "checkpoint_format_version": 1,
            "epoch": 5,
        }
        safe_atomic_save(payload, ckpt_file)

        contract = get_verification_contract("nafnet")
        res, _loaded_model, loaded_path = self.engine.verify_loadability(
            model_key="nafnet",
            checkpoint_path=ckpt_file,
            contract=contract,
            model=model,
        )

        self.assertTrue(res.passed)
        self.assertEqual(res.stage, VerificationStage.LOADABILITY)
        self.assertEqual(loaded_path, ckpt_file.resolve())

    def test_loadability_fails_missing_keys(self) -> None:
        """Test loadability fails when required state dict keys are absent."""
        ckpt_file = self.temp_path / "corrupt_model.pth"
        payload = {"invalid_key": 123, "checkpoint_format_version": 1}
        safe_atomic_save(payload, ckpt_file)

        contract = get_verification_contract("nafnet")
        res, _, _ = self.engine.verify_loadability(
            model_key="nafnet",
            checkpoint_path=ckpt_file,
            contract=contract,
            model=SingleConvModel(),
        )

        self.assertFalse(res.passed)
        self.assertIn("Missing mandatory checkpoint payload key", res.error or "")

    def test_determinism_verification(self) -> None:
        """Test forward determinism verification passing on deterministic model and failing on noisy model."""
        contract = get_verification_contract("nafnet")
        dummy = torch.randn(1, 3, 32, 32)

        # 1. Deterministic model passes
        clean_model = SingleConvModel()
        res_clean = self.engine.verify_determinism(clean_model, dummy, contract)
        self.assertTrue(res_clean.passed)
        self.assertLessEqual(res_clean.details.get("max_abs_diff", 1.0), 1e-6)

        # 2. Non-deterministic model fails
        noisy_model = NonDeterministicModel()
        res_noisy = self.engine.verify_determinism(noisy_model, dummy, contract)
        self.assertFalse(res_noisy.passed)
        self.assertIn("non-deterministic", res_noisy.error or "")

    def test_metric_thresholds_verification(self) -> None:
        """Test validation metric thresholds checking."""
        contract = get_verification_contract("nafnet")

        # Passing metrics
        good_metrics = {"psnr": 25.5, "ssim": 0.85, "lpips": 0.15, "fid": 20.0}
        res_good = self.engine.verify_metric_thresholds(good_metrics, contract)
        self.assertTrue(res_good.passed)

        # Failing metrics (lower than min psnr)
        bad_metrics = {"psnr": 15.0, "ssim": 0.50}
        res_bad = self.engine.verify_metric_thresholds(bad_metrics, contract)
        self.assertFalse(res_bad.passed)
        self.assertIn("psnr", res_bad.error or "")

    def test_full_verification_tiny_model(self) -> None:
        """Test full multi-stage verification pipeline on SingleConvModel."""
        model = SingleConvModel()
        ckpt_file = self.temp_path / "tiny_model.pth"
        payload = {
            "model_state_dict": model.state_dict(),
            "checkpoint_format_version": 1,
            "epoch": 1,
        }
        safe_atomic_save(payload, ckpt_file)

        report_file = self.temp_path / "verification_report.json"
        report = self.engine.run_full_verification(
            model_key="nafnet",
            checkpoint_path=ckpt_file,
            model=model,
            dummy_input=torch.randn(1, 3, 32, 32),
            metrics={"psnr": 28.0, "ssim": 0.90},
            hub_dir=self.temp_path,
            temp_dir=self.temp_path,
        )

        self.assertTrue(report.overall_passed)
        self.assertEqual(len(report.errors), 0)
        self.assertIn("loadability", report.stages)
        self.assertIn("determinism", report.stages)
        self.assertIn("metric_thresholds", report.stages)
        self.assertIn("export_equivalence", report.stages)
        self.assertIn("schema_conformance", report.stages)

        # Verify JSON report generation
        with report_file.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f)

        self.assertTrue(report_file.exists())
        loaded_json = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded_json["model_key"], "nafnet")
        self.assertTrue(loaded_json["overall_passed"])


if __name__ == "__main__":
    unittest.main()
