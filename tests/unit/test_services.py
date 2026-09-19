"""Unit tests for LemGendary Services Layer (Gate 11).

Verifies in-process S.O.L.I.D. services: TrainingService, EvaluationService, CheckpointService,
ExportService, CloudSyncService, NotebookService, and AuditService.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import unittest
import torch
import yaml

from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.eval_service import EvaluationService
from training.services.export_service import ExportService
from training.services.notebook_service import NotebookService
from training.services.sync_service import CloudSyncService
from training.services.training_service import TrainingService


class TestServicesLayer(unittest.TestCase):
    """Test suite for S.O.L.I.D. in-process services."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create dummy presets.yaml
        presets_data = {
            "presets": {
                "test-preset": {
                    "description": "Test Preset Description",
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                },
                "quick-sota": {
                    "description": "Quick SOTA run",
                    "epochs": 10,
                    "batch_size": 8,
                    "learning_rate": 0.0005,
                },
            }
        }
        with open(self.root_path / "presets.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(presets_data, f)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_training_service_presets(self) -> None:
        """Verify TrainingService correctly parses and queries presets."""
        service = TrainingService(project_root=self.root_path)
        presets = service.load_presets()
        self.assertIn("test-preset", presets)
        self.assertIn("quick-sota", presets)

        preset = service.get_preset("test-preset")
        self.assertIsNotNone(preset)
        self.assertEqual(preset["epochs"], 5)
        self.assertEqual(preset["batch_size"], 4)
        self.assertEqual(preset["learning_rate"], 0.001)

        missing = service.get_preset("nonexistent")
        self.assertIsNone(missing)

    def test_checkpoint_service_inspect_and_prune(self) -> None:
        """Verify CheckpointService listing, inspection, and pruning."""
        checkpoints_dir = self.root_path / "checkpoints" / "test_model"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy checkpoints
        for epoch in [1, 2, 3, 4, 5]:
            ckpt_path = checkpoints_dir / f"test_model_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": 0.5 / epoch,
                    "metrics": {"val_loss": 0.5 / epoch},
                    "model_state": {"weight": torch.ones(2, 2)},
                },
                ckpt_path,
            )

        best_path = checkpoints_dir / "test_model_best.pth"
        torch.save(
            {
                "epoch": 5,
                "best_loss": 0.1,
                "metrics": {"val_loss": 0.1},
                "model_state": {"weight": torch.ones(2, 2)},
            },
            best_path,
        )

        service = CheckpointService(project_root=self.root_path)
        listed = service.list_checkpoints(model_key="test_model")
        self.assertEqual(len(listed), 6)

        # Test inspection
        info = service.inspect_checkpoint(best_path)
        self.assertEqual(info["epoch"], 5)
        self.assertEqual(info["parameter_count"], 4)
        self.assertIn("model_state", info["keys"])

        # Test pruning (keep last 2, plus best)
        pruned = service.prune_checkpoints("test_model", keep_last_k=2, keep_best=True)
        self.assertEqual(len(pruned), 3)  # epochs 1, 2, 3 pruned
        self.assertTrue((checkpoints_dir / "test_model_epoch_4.pth").exists())
        self.assertTrue((checkpoints_dir / "test_model_epoch_5.pth").exists())
        self.assertTrue(best_path.exists())

    def test_export_service_targets(self) -> None:
        """Verify ExportService supported targets and error handling."""
        service = ExportService(project_root=self.root_path)
        targets = service.list_supported_targets()
        self.assertIn("onnx_fp32", targets)
        self.assertIn("onnx_fp16", targets)
        self.assertIn("torch_pt", targets)
        self.assertIn("webgpu", targets)
        self.assertIn("forex", targets)

        with self.assertRaises(ValueError):
            service.export("test_model", targets=["invalid_target"])

    def test_cloud_sync_service(self) -> None:
        """Verify CloudSyncService provider probe and dry run."""
        service = CloudSyncService(project_root=self.root_path)
        health = service.probe_providers()
        self.assertIn("gdrive", health)
        self.assertIn("kaggle", health)
        self.assertIn("github", health)

        # Invalid target raises error
        with self.assertRaises(ValueError):
            service.sync_model("test_model", target="unsupported")

        # Dry run on non-existent folder
        res = service.sync_model("nonexistent_model", target="gdrive", dry_run=True)
        self.assertFalse(res["success"])

    def test_notebook_service(self) -> None:
        """Verify NotebookService validation and listing."""
        service = NotebookService(project_root=self.root_path)
        models = service.list_supported_models()
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0)

        with self.assertRaises(ValueError):
            service.generate_notebooks("nima_aesthetic_mobile", platform="invalid_platform")

        with self.assertRaises(ValueError):
            service.generate_notebooks("nima_aesthetic_mobile", platform="kaggle", kinds=["invalid_kind"])

    def test_audit_service_system_and_model(self) -> None:
        """Verify AuditService system resource and model topology auditing."""
        service = AuditService()
        sys_info = service.audit_system()
        self.assertIn("platform", sys_info)
        self.assertIn("python_version", sys_info)
        self.assertIn("torch_version", sys_info)
        self.assertIn("cpu_threads", sys_info)
        self.assertIn("disk_healthy", sys_info)
        self.assertTrue(sys_info["cpu_threads"] >= 1)

        model_info = service.audit_model("forex_predictor")
        self.assertEqual(model_info["model_key"], "forex_predictor")
        self.assertTrue(model_info["total_parameters"] > 0)
        self.assertTrue(model_info["trainable_parameters"] > 0)
        self.assertTrue(model_info["estimated_fp32_mb"] > 0)


if __name__ == "__main__":
    unittest.main()
