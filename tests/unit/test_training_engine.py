"""Unit tests for LemGendary Core Training Engine Subpackage.

Verifies TrainingContext, AMP helpers, optimizer/scheduler builders,
epoch runner, validation passes, and the central run_training coordinator (Gate 10).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpoint import MetricVault, ResumeState
from training.governance import SmartTrainingGovernor, SotaTracker
from training.hardware.discovery import DeviceInfo, discover_device
from training.hardware.policy import ExecutionPolicy
from training.hardware.sentinel import SentinelGuard
from training.training import (
    TrainingContext,
    TrainingPaths,
    build_optimizer,
    build_scheduler,
    create_grad_scaler,
    get_autocast_context,
    run_training,
    safe_backward,
    train_one_epoch,
    validate_one_epoch,
)


class SimpleModel(nn.Module):
    """Simple linear regression model for unit testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TestTrainingEngine(unittest.TestCase):
    """Test suite for modular training engine subpackage."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.device_info = discover_device(force_device="cpu")
        self.policy = ExecutionPolicy(
            amp_enabled=False,
            amp_dtype=torch.float32,
            cudnn_benchmark=False,
            tf32_enabled=False,
            channels_last=False,
            scaler=None,
        )

    def test_optimizer_and_scheduler_builders(self) -> None:
        """Verify optimizer and scheduler factories construct proper parameter groups and schedules."""
        model = SimpleModel()
        config = {
            "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.05},
            "scheduler": {"type": "cosine", "min_lr": 1e-6},
        }
        optimizer = build_optimizer(model, config)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.05)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        scheduler = build_scheduler(optimizer, config, total_epochs=10)
        self.assertIsNotNone(scheduler)

    def test_amp_and_safe_backward(self) -> None:
        """Verify safe_backward computes backward pass and calculates gradient norm."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 8)
        y = torch.randn(4, 1)

        out = model(x)
        loss = nn.functional.mse_loss(out, y)

        grad_norm = safe_backward(loss, optimizer, scaler=None, max_norm=1.0, model=model)
        self.assertGreaterEqual(grad_norm, 0.0)

    def test_train_and_validate_one_epoch(self) -> None:
        """Verify train_one_epoch and validate_one_epoch return accurate telemetry metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            criterion = nn.MSELoss()

            x_train = torch.randn(16, 8)
            y_train = torch.randn(16, 1)
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4, shuffle=True)

            x_val = torch.randn(8, 8)
            y_val = torch.randn(8, 1)
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)

            paths = TrainingPaths(
                project_root=tmp_path,
                local_checkpoint_dir=tmp_path / "checkpoints",
                hub_checkpoint_dir=tmp_path / "hub",
                export_dir=tmp_path / "export",
                progress_local_path=tmp_path / "progress.pth",
                best_checkpoint_path=tmp_path / "best.pth",
                history_csv_path=tmp_path / "history.csv",
            )

            sentinel = SentinelGuard(device=self.device)
            governor = SmartTrainingGovernor(model_info={})
            sota_tracker = SotaTracker()
            vault = MetricVault()

            ctx = TrainingContext(
                model_name="test_model",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device_info=self.device_info,
                policy=self.policy,
                sentinel=sentinel,
                governor=governor,
                sota_tracker=sota_tracker,
                vault=vault,
                paths=paths,
                config={},
                model_info={},
                args=None,
                total_epochs=2,
            )

            train_metrics = train_one_epoch(ctx, 1)
            self.assertIn("train_loss", train_metrics)
            self.assertIn("grad_norm", train_metrics)
            self.assertIn("lr", train_metrics)

            val_metrics = validate_one_epoch(ctx, 1)
            self.assertIn("val_loss", val_metrics)

    def test_run_training_gate_10(self) -> None:
        """Gate 10: Execute a complete multi-epoch mock training session."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            criterion = nn.MSELoss()

            x_train = torch.randn(12, 8)
            y_train = torch.randn(12, 1)
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)

            x_val = torch.randn(4, 8)
            y_val = torch.randn(4, 1)
            val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)

            paths = TrainingPaths(
                project_root=tmp_path,
                local_checkpoint_dir=tmp_path / "checkpoints",
                hub_checkpoint_dir=tmp_path / "hub",
                export_dir=tmp_path / "export",
                progress_local_path=tmp_path / "progress.pth",
                best_checkpoint_path=tmp_path / "best.pth",
                history_csv_path=tmp_path / "history.csv",
            )
            paths.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            paths.export_dir.mkdir(parents=True, exist_ok=True)

            sentinel = SentinelGuard(device=self.device)
            governor = SmartTrainingGovernor(model_info={})
            sota_tracker = SotaTracker()
            vault = MetricVault()

            ctx = TrainingContext(
                model_name="test_model",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                device_info=self.device_info,
                policy=self.policy,
                sentinel=sentinel,
                governor=governor,
                sota_tracker=sota_tracker,
                vault=vault,
                paths=paths,
                config={},
                model_info={},
                args=None,
                total_epochs=2,
            )

            summary = run_training(ctx)

            # Assertions
            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.final_epoch, 2)
            self.assertTrue(paths.progress_local_path.exists())
            self.assertTrue(paths.best_checkpoint_path.exists())
            self.assertTrue(len(summary.best_metrics) > 0)


if __name__ == "__main__":
    unittest.main()
