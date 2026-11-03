"""Stochastic Weight Averaging: config, averaging math, BatchNorm refresh, resume, soup.

CPU-only with tiny models. Covers the ``swa`` config section, ``SwaAverager``
(``AveragedModel``/``SWALR``/``update_bn``), the ``swa_state.pth`` resume
sidecar, the engine seam (a disabled run is unchanged) and ``vault_swa.pth`` as a
Pareto Model Soup ingredient.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.checkpoint import MetricVault, ResumeState
from training.governance import SmartTrainingGovernor, SotaTracker
from training.hardware.discovery import discover_device
from training.hardware.policy import ExecutionPolicy
from training.hardware.sentinel import SentinelGuard
from training.model_soup import blend_model_soup
from training.swa_config import SwaConfig, SwaLrSchedule, SwaStart
from training.training import TrainingContext, TrainingPaths, run_training, train_one_epoch
from training.training.swa import (
    SWA_STATE_FILENAME,
    SWA_VAULT_FILENAME,
    SwaAverager,
    SwaPhase,
)


class TinyNet(nn.Module):
    """Two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear, ReLU, linear."""
        return self.fc2(torch.relu(self.fc1(x)))


class TinyBnNet(nn.Module):
    """Linear layer followed by BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BatchNorm only."""
        return self.bn(x)


def _fill(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class TestSwaConfig(unittest.TestCase):
    """The ``swa`` config section."""

    def test_missing_section_is_disabled(self) -> None:
        """No section means SWA is off with documented defaults."""
        cfg = SwaConfig.from_config({})
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.lr)
        self.assertEqual(SwaConfig.from_config(None), cfg)

    def test_start_epoch_and_fraction(self) -> None:
        """``start_epoch`` wins over ``start_fraction``; the fraction averages the tail."""
        self.assertEqual(SwaConfig(start=SwaStart(fraction=0.75)).first_epoch(100), 76)
        self.assertEqual(SwaConfig(start=SwaStart(fraction=0.0)).first_epoch(10), 1)
        self.assertEqual(SwaConfig(start=SwaStart(epoch=7, fraction=0.5)).first_epoch(100), 7)

    def test_invalid_values_raise(self) -> None:
        """Wrong types and out-of-range values are rejected."""
        bad: list[tuple[dict[str, Any], type[Exception]]] = [
            ({"swa": ["enabled"]}, TypeError),
            ({"swa": {"enabled": "yes"}}, TypeError),
            ({"swa": {"start_fraction": 1.0}}, ValueError),
            ({"swa": {"swa_lr": 0}}, ValueError),
            ({"swa": {"swa_lr": "fast"}}, TypeError),
            ({"swa": {"anneal_strategy": "step"}}, ValueError),
            ({"swa": {"update_every": 0}}, ValueError),
            ({"swa": {"start_epoch": 0}}, ValueError),
            ({"swa": {"anneal_epochs": 1.5}}, TypeError),
        ]
        for config, error in bad:
            with self.subTest(config=config), self.assertRaises(error):
                SwaConfig.from_config(config)


class TestSwaAverager(unittest.TestCase):
    """``AveragedModel``, ``SWALR`` and ``update_bn`` behind ``SwaAverager``."""

    def test_running_mean_of_weights(self) -> None:
        """Three snapshots give their arithmetic mean; the live model is untouched."""
        model = TinyNet()
        averager = SwaAverager(model, torch.optim.SGD(model.parameters(), lr=0.1), SwaConfig())
        self.assertIsNone(averager.module)
        for value in (1.0, 2.0, 6.0):
            _fill(model, value)
            averager.update()
        self.assertEqual(averager.n_averaged, 3)
        assert averager.module is not None
        for param in averager.module.parameters():
            torch.testing.assert_close(param, torch.full_like(param, 3.0))
        for param in model.parameters():
            torch.testing.assert_close(param, torch.full_like(param, 6.0))

    def test_swalr_anneals_to_swa_lr(self) -> None:
        """With ``swa_lr`` set the LR reaches it after ``anneal_epochs`` steps."""
        model = TinyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        averager = SwaAverager(
            model,
            optimizer,
            SwaConfig(lr=SwaLrSchedule(0.01, anneal_epochs=2, anneal_strategy="linear")),
        )
        averager.step_lr()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.055)
        averager.step_lr()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
        averager.step_lr()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_no_swa_lr_leaves_lr_alone(self) -> None:
        """Without ``swa_lr`` no SWALR is created."""
        model = TinyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        averager = SwaAverager(model, optimizer, SwaConfig())
        averager.step_lr()
        self.assertIsNone(averager.scheduler)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_refresh_bn_recomputes_running_stats(self) -> None:
        """``update_bn`` sets the averaged BatchNorm stats to the data statistics."""
        torch.manual_seed(0)
        model = TinyBnNet()
        averager = SwaAverager(model, torch.optim.SGD(model.parameters(), lr=0.1), SwaConfig())
        averager.update()
        data = torch.randn(64, 3) * 2.0 + 5.0
        loader = DataLoader(TensorDataset(data, torch.zeros(64, 1)), batch_size=16)
        averager.refresh_bn(loader, torch.device("cpu"))
        assert averager.module is not None
        bn = averager.module.bn
        torch.testing.assert_close(bn.running_mean, data.mean(dim=0), atol=1e-4, rtol=1e-4)
        # The live model's statistics are not touched.
        torch.testing.assert_close(model.bn.running_mean, torch.zeros(3))

    def test_refresh_bn_batch_cap(self) -> None:
        """``bn_update_batches`` limits the refresh pass."""
        model = TinyBnNet()
        config = SwaConfig(bn_update_batches=1)
        averager = SwaAverager(model, torch.optim.SGD(model.parameters(), lr=0.1), config)
        averager.update()
        first = torch.full((4, 3), 1.0)
        rest = torch.full((4, 3), 9.0)
        loader = DataLoader(
            TensorDataset(torch.cat([first, rest]), torch.zeros(8, 1)), batch_size=4
        )
        averager.refresh_bn(loader, torch.device("cpu"))
        assert averager.module is not None
        torch.testing.assert_close(averager.module.bn.running_mean, torch.ones(3))

    def test_state_dict_round_trip(self) -> None:
        """Averaged weights, ``n_averaged`` and the SWALR position survive a round trip."""
        config = SwaConfig(lr=SwaLrSchedule(0.01, anneal_epochs=4))
        model = TinyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        averager = SwaAverager(model, optimizer, config)
        for value in (1.0, 3.0):
            _fill(model, value)
            averager.update()
            averager.step_lr()
        state = averager.state_dict()

        model2 = TinyNet()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        optimizer2.load_state_dict(optimizer.state_dict())
        restored = SwaAverager(model2, optimizer2, config)
        restored.load_state_dict(state)
        self.assertEqual(restored.n_averaged, 2)
        assert restored.module is not None and averager.module is not None
        for key, value in averager.module.state_dict().items():
            torch.testing.assert_close(restored.module.state_dict()[key], value)
        averager.step_lr()
        restored.step_lr()
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"], optimizer2.param_groups[0]["lr"], places=12
        )


_CPU_POLICY = ExecutionPolicy(
    amp_enabled=False,
    amp_dtype=torch.float32,
    cudnn_benchmark=False,
    tf32_enabled=False,
    channels_last=False,
    scaler=None,
)


def _build_ctx(root: Path, config: dict[str, Any], total_epochs: int = 3) -> TrainingContext:
    """A deterministic context for ``TinyNet`` on fixed random data under ``root``."""
    torch.manual_seed(1234)
    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    x_train, y_train = torch.randn(12, 8), torch.randn(12, 1)
    x_val, y_val = torch.randn(4, 8), torch.randn(4, 1)
    paths = TrainingPaths(
        project_root=root,
        local_checkpoint_dir=root / "checkpoints",
        hub_checkpoint_dir=root / "hub",
        export_dir=root / "export",
        progress_local_path=root / "progress.pth",
        best_checkpoint_path=root / "best.pth",
        history_csv_path=root / "history.csv",
    )
    paths.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.export_dir.mkdir(parents=True, exist_ok=True)
    return TrainingContext(
        model_name="swa_test_model",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.MSELoss(),
        train_loader=DataLoader(TensorDataset(x_train, y_train), batch_size=4),
        val_loader=DataLoader(TensorDataset(x_val, y_val), batch_size=4),
        device_info=discover_device(force_device="cpu"),
        policy=_CPU_POLICY,
        sentinel=SentinelGuard(device=torch.device("cpu")),
        governor=SmartTrainingGovernor(model_info={}),
        sota_tracker=SotaTracker(),
        vault=MetricVault(),
        paths=paths,
        config=config,
        model_info={},
        args=None,
        total_epochs=total_epochs,
    )


class TestSwaEngineSeam(unittest.TestCase):
    """SWA wired into ``run_training``."""

    def _run(self, config: dict[str, Any]) -> tuple[TrainingContext, list[Any]]:
        """Run three epochs; return the context and the weights after every epoch."""
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        ctx = _build_ctx(root, config)
        snapshots: list[dict[str, torch.Tensor]] = []
        run_training(ctx, on_epoch_end=lambda _epoch, _m: snapshots.append(_state(ctx.model)))
        return ctx, snapshots

    def test_disabled_run_is_unchanged(self) -> None:
        """No section, ``enabled: false`` and a never-reached start give identical runs."""
        base, base_snaps = self._run({})
        self.assertIsNone(base.swa)
        variants = [
            {"swa": {"enabled": False, "swa_lr": 0.5}},
            {"swa": {"enabled": True, "start_epoch": 99, "swa_lr": 0.5}},
        ]
        for config in variants:
            with self.subTest(config=config):
                ctx, snaps = self._run(config)
                self.assertEqual(len(snaps), len(base_snaps))
                for got, want in zip(snaps, base_snaps):
                    for key, value in want.items():
                        torch.testing.assert_close(got[key], value, rtol=0, atol=0)
                self.assertEqual(
                    ctx.optimizer.param_groups[0]["lr"], base.optimizer.param_groups[0]["lr"]
                )
                ckpt_dir = ctx.lifecycle_manager.checkpoints_dir
                self.assertFalse((ckpt_dir / SWA_STATE_FILENAME).exists())
                self.assertFalse((ckpt_dir / SWA_VAULT_FILENAME).exists())
                self.assertNotIn("swa_metrics", ctx.metadata)

    def test_enabled_run_averages_and_publishes(self) -> None:
        """Epochs 2..3 are averaged, evaluated and saved as sidecar plus soup ingredient."""
        ctx, snaps = self._run({"swa": {"enabled": True, "start_epoch": 2}})
        ckpt_dir = ctx.lifecycle_manager.checkpoints_dir
        ingredient = torch.load(ckpt_dir / SWA_VAULT_FILENAME, weights_only=False)
        self.assertTrue(ingredient["is_swa"])
        self.assertEqual(ingredient["swa_n_averaged"], 2)
        self.assertEqual(ingredient["vault_metric"], "swa")
        for key, value in ingredient["model_state"].items():
            expected = (snaps[1][key] + snaps[2][key]) / 2.0
            torch.testing.assert_close(value, expected)
        self.assertIn("val_loss", ctx.metadata["swa_metrics"])
        sidecar = torch.load(ckpt_dir / SWA_STATE_FILENAME, weights_only=False)
        self.assertEqual(sidecar["epoch"], 3)
        self.assertEqual(sidecar["start_epoch"], 2)
        # The live model is not replaced by the average.
        for key, value in snaps[2].items():
            torch.testing.assert_close(ctx.model.state_dict()[key], value, rtol=0, atol=0)

    def test_swalr_owns_lr_in_phase(self) -> None:
        """In the SWA phase the main scheduler is not stepped; SWALR sets the LR."""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _build_ctx(
                Path(tmp),
                {"swa": {"enabled": True, "start_epoch": 1, "swa_lr": 0.001, "anneal_epochs": 0}},
            )
            run_training(ctx)
            self.assertAlmostEqual(ctx.optimizer.param_groups[0]["lr"], 0.001)
            self.assertEqual(ctx.scheduler.last_epoch, 0)

    def test_owns_lr_skips_main_scheduler_in_train_pass(self) -> None:
        """``train_one_epoch`` steps the main scheduler only outside the SWALR phase."""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _build_ctx(Path(tmp), {})
            config = SwaConfig(enabled=True, lr=SwaLrSchedule(0.01))
            ctx.swa = SwaPhase(config, SwaAverager(ctx.model, ctx.optimizer, config), 2, None)
            train_one_epoch(ctx, 1)
            self.assertEqual(ctx.scheduler.last_epoch, 1)
            train_one_epoch(ctx, 2)
            self.assertEqual(ctx.scheduler.last_epoch, 1)


class TestSwaResume(unittest.TestCase):
    """The ``swa_state.pth`` sidecar."""

    def _phase(self, ckpt_dir: Path) -> tuple[SwaPhase, nn.Module]:
        model = TinyNet()
        config = SwaConfig(enabled=True, lr=SwaLrSchedule(0.01, anneal_epochs=3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return SwaPhase(config, SwaAverager(model, optimizer, config), 2, ckpt_dir), model

    def _write(self, ckpt_dir: Path) -> SwaPhase:
        phase, model = self._phase(ckpt_dir)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _build_ctx(Path(tmp), {})
            for epoch, value in ((2, 1.0), (3, 2.0)):
                _fill(model, value)
                phase.end_epoch(ctx, epoch)
        return phase

    @staticmethod
    def _resume(epoch: int, intra: bool = False) -> ResumeState:
        return ResumeState(epoch=epoch, iteration=0, model_state_dict={}, is_intra_epoch=intra)

    def test_round_trip_at_completed_epoch(self) -> None:
        """Resuming after epoch 3 (or inside epoch 4) restores the average."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            saved = self._write(ckpt_dir)
            for resume in (self._resume(3), self._resume(4, intra=True)):
                with self.subTest(resume=resume):
                    phase, _ = self._phase(ckpt_dir)
                    phase.start_epoch = 50
                    phase.restore(resume)
                    self.assertEqual(phase.start_epoch, 2)
                    self.assertEqual(phase.averager.n_averaged, 2)
                    assert phase.averager.module is not None
                    assert saved.averager.module is not None
                    for param in phase.averager.module.parameters():
                        torch.testing.assert_close(param, torch.full_like(param, 1.5))
                    self.assertIsNotNone(phase.averager.scheduler)

    def test_mismatched_epoch_restarts(self) -> None:
        """A sidecar from another epoch is ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            self._write(ckpt_dir)
            phase, _ = self._phase(ckpt_dir)
            with self.assertLogs("lemtrain.training.swa", level="WARNING"):
                phase.restore(self._resume(2))
            self.assertEqual(phase.averager.n_averaged, 0)
            self.assertTrue((ckpt_dir / SWA_STATE_FILENAME).exists())

    def test_fresh_run_removes_stale_sidecar(self) -> None:
        """Without a resume state the old sidecar is deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            self._write(ckpt_dir)
            phase, _ = self._phase(ckpt_dir)
            phase.restore(None)
            self.assertEqual(phase.averager.n_averaged, 0)
            self.assertFalse((ckpt_dir / SWA_STATE_FILENAME).exists())

    def test_corrupt_sidecar_is_ignored(self) -> None:
        """An unreadable sidecar logs a warning and averaging restarts."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            (ckpt_dir / SWA_STATE_FILENAME).write_bytes(b"not a checkpoint")
            phase, _ = self._phase(ckpt_dir)
            with self.assertLogs("lemtrain.training.swa", level="WARNING"):
                phase.restore(self._resume(3))
            self.assertEqual(phase.averager.n_averaged, 0)


class TestSwaSoupIngredient(unittest.TestCase):
    """``vault_swa.pth`` blends with the other vault checkpoints."""

    def test_swa_vault_is_soup_ingredient(self) -> None:
        """The soup service picks up ``vault_swa.pth`` by name and by default glob."""
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "suite"
            project_root.mkdir()
            ckpt_dir = Path(tmp) / "LemGendaryModels" / "swa_test_model" / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            with tempfile.TemporaryDirectory() as run_tmp:
                ctx = _build_ctx(Path(run_tmp), {})
                model = ctx.model
                config = SwaConfig(enabled=True)
                phase = SwaPhase(config, SwaAverager(model, ctx.optimizer, config), 1, ckpt_dir)
                for epoch, value in ((1, 2.0), (2, 4.0)):
                    _fill(model, value)
                    phase.end_epoch(ctx, epoch)
                phase.finish(ctx, 2)
            self.assertTrue((ckpt_dir / SWA_VAULT_FILENAME).exists())

            _fill(model, 1.0)
            torch.save({"model_state": _state(model), "epoch": 1}, ckpt_dir / "vault_psnr.pth")

            out = blend_model_soup("swa_test_model", project_root, donors=["psnr", "swa"])
            soup = torch.load(out.output_path, weights_only=False)
            self.assertEqual(soup["soup_donors"], ["vault_psnr.pth", "vault_swa.pth"])
            for value in soup["model_state"].values():
                torch.testing.assert_close(value, torch.full_like(value, 2.0))

            default = blend_model_soup("swa_test_model", project_root, output="all.pth")
            self.assertIn(SWA_VAULT_FILENAME, [p.name for p in default.donors])


if __name__ == "__main__":
    unittest.main()
