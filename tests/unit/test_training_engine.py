"""Unit tests for LemGendary Core Training Engine Subpackage.

Verifies TrainingContext, AMP helpers, optimizer/scheduler builders,
epoch runner, validation passes, and the central run_training coordinator (Gate 10).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests.fixtures.training_context import fresh_governance
from training.hardware.discovery import discover_device
from training.hardware.policy import ExecutionPolicy
from training.hardware.sentinel import SentinelGuard
from training.training import (
    DataComponents,
    GradClip,
    HardwareComponents,
    ModelStack,
    RunSettings,
    TrainingContext,
    TrainingPaths,
    build_optimizer,
    build_scheduler,
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
        """Apply the two linear layers with a ReLU in between."""
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
        """Verify optimizer/scheduler factories build proper parameter groups and schedules."""
        model = SimpleModel()
        config = {
            "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.05},
            "scheduler": {"type": "cosine", "min_lr": 1e-6},
        }
        optimizer = build_optimizer(model, config)
        # Backbone decay/no-decay, then head decay/no-decay (the last child is the head).
        self.assertEqual(
            [(g["group"], g["weight_decay"]) for g in optimizer.param_groups],
            [("backbone", 0.05), ("backbone", 0.0), ("head", 0.05), ("head", 0.0)],
        )

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

        grad_norm = safe_backward(
            loss, optimizer, scaler=None, clip=GradClip(model=model, max_norm=1.0)
        )
        self.assertGreaterEqual(grad_norm, 0.0)

    def _make_context(
        self, tmp_path: Path, model: nn.Module, optimizer, loaders: tuple
    ) -> TrainingContext:
        """Build a two-epoch MSE ``TrainingContext`` rooted at ``tmp_path``."""
        paths = TrainingPaths(
            project_root=tmp_path,
            local_checkpoint_dir=tmp_path / "checkpoints",
            export_dir=tmp_path / "export",
            progress_local_path=tmp_path / "progress.pth",
            best_checkpoint_path=tmp_path / "best.pth",
            history_csv_path=tmp_path / "history.csv",
        )
        return TrainingContext(
            model_name="test_model",
            models=ModelStack(
                model=model,
                optimizer=optimizer,
                scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9),
                criterion=nn.MSELoss(),
            ),
            data=DataComponents(
                train_loader=loaders[0],
                val_loader=loaders[1],
            ),
            hardware=HardwareComponents(
                device_info=self.device_info,
                policy=self.policy,
                sentinel=SentinelGuard(device=self.device),
            ),
            governance=fresh_governance({}),
            run=RunSettings(
                paths=paths,
                config={},
                model_info={},
                args=None,
                total_epochs=2,
            ),
        )

    def test_train_and_validate_one_epoch(self) -> None:
        """Verify train_one_epoch and validate_one_epoch return accurate telemetry metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loaders = (_tensor_loader(16, shuffle=True), _tensor_loader(8))
            ctx = self._make_context(Path(tmp_dir), model, optimizer, loaders)

            train_metrics = train_one_epoch(ctx, 1)
            self.assertIn("train_loss", train_metrics)
            self.assertIn("grad_norm", train_metrics)
            self.assertIn("lr", train_metrics)

            val_metrics = validate_one_epoch(ctx, 1)
            self.assertIn("val_loss", val_metrics)

    def test_run_training_gate_10(self) -> None:
        """Gate 10: Execute a complete multi-epoch mock training session."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loaders = (_tensor_loader(12), _tensor_loader(4))
            ctx = self._make_context(Path(tmp_dir), model, optimizer, loaders)
            ctx.run.paths.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ctx.run.paths.export_dir.mkdir(parents=True, exist_ok=True)

            summary = run_training(ctx)

            # Assertions
            self.assertTrue(ctx.governance.lifecycle_manager.best_path.exists())
            self.assertTrue(ctx.governance.lifecycle_manager.metrics_csv_path.exists())
            self.assertTrue(len(summary.best_metrics) > 0)

    def test_validate_one_epoch_forex(self) -> None:
        """Verify validate_one_epoch computes direction and pip-regression metrics for Forex."""

        class MockForexModel(nn.Module):
            """Forex model stub that always predicts BUY with fixed magnitudes."""

            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 3)

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                """Return constant BUY direction logits and magnitude predictions."""
                bs = x.shape[0]
                dir_logits = torch.zeros(bs, 3)
                dir_logits[:, 2] = 2.0  # Predict BUY
                mag_preds = torch.full((bs, 2), 15.0)
                return {
                    "direction_logits": dir_logits,
                    "magnitude": mag_preds,
                }

        model = MockForexModel()

        def criterion(_preds, _targets):
            return torch.tensor(0.25)

        x_val = torch.randn(4, 8)
        targets = {
            "direction": torch.tensor([2, 2, 0, 1], dtype=torch.long),
            "magnitude": torch.full((4, 2), 15.0),
        }

        class MockForexValLoader:
            """Validation loader stub yielding a single forex batch."""

            def __iter__(self):
                yield x_val, targets

            def __len__(self):
                return 1

        val_loader = MockForexValLoader()
        ctx = TrainingContext(
            model_name="forex_predictor",
            models=ModelStack(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
                scheduler=None,
                criterion=criterion,
            ),
            data=DataComponents(
                train_loader=None,
                val_loader=val_loader,
            ),
            hardware=HardwareComponents(
                device_info=self.device_info,
                policy=self.policy,
                sentinel=SentinelGuard(device=self.device),
            ),
            governance=fresh_governance({"category": "forex"}),
            run=RunSettings(
                paths=TrainingPaths(
                    project_root=Path("."),
                    local_checkpoint_dir=Path("."),
                    export_dir=Path("."),
                    progress_local_path=Path("."),
                    best_checkpoint_path=Path("."),
                    history_csv_path=Path("."),
                ),
                config={},
                model_info={"category": "forex"},
                args=None,
                total_epochs=1,
            ),
        )

        val_metrics = validate_one_epoch(ctx, 1)
        self.assertIn("dir_acc", val_metrics)
        self.assertIn("win_rate", val_metrics)
        self.assertIn("profit_factor", val_metrics)
        self.assertIn("sharpe_ratio", val_metrics)
        self.assertIn("sortino_ratio", val_metrics)
        self.assertIn("max_drawdown", val_metrics)
        self.assertIn("tp_mae", val_metrics)
        self.assertIn("sl_mae", val_metrics)
        self.assertIn("dir_entropy", val_metrics)
        self.assertIn("quality_score", val_metrics)
        self.assertEqual(val_metrics["dir_acc"], 50.0)


def _tensor_loader(n: int, shuffle: bool = False) -> DataLoader:
    """A batch-4 loader over ``n`` random (8-feature, 1-target) pairs."""
    return DataLoader(
        TensorDataset(torch.randn(n, 8), torch.randn(n, 1)), batch_size=4, shuffle=shuffle
    )


if __name__ == "__main__":
    unittest.main()
