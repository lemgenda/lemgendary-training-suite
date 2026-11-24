"""Unit tests for checkpoint management, recovery engine, resume scaling, and metric vault."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointLifecycleManager,
    CheckpointMigrationError,
    CheckpointRecoveryEngine,
    MetricVault,
    ResumeState,
    audit_disk_space,
    parse_resume_state,
    prune_vault_checkpoints,
    safe_atomic_save,
    safe_load_checkpoint,
    scale_resume_progress,
    stretch_scheduler_runway,
    validate_vault_consistency,
)
from training.training.step_progress import resume_iteration
from training.utils.rng_state import restore_rng_state, rng_payload


class TestCheckpointSubsystem(unittest.TestCase):
    """Test suite covering atomic checkpoint saves, recovery, resume scaling, and metric vault."""

    def setUp(self) -> None:
        # The recovery engine returns canonical (resolved) roots; on Windows the
        # temp dir may be an 8.3 short path (e.g. ``SANDI~1``), so the fixture
        # root must be canonical too for path-equality assertions to hold.
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root)

    def isolated_project_root(self) -> Path:
        """Project root nested in the temp dir, so the models hub resolves inside it.

        ``get_models_hub_root`` prefers ``<root>.parent / "LemGendaryModels"``; for a root
        directly under the system temp dir that is a hub shared by every test run, whose
        stale checkpoints made results depend on earlier runs.
        """
        return self.root / "suite"

    def test_atomic_save_and_load(self) -> None:
        """Verify atomic checkpoint save writes a payload that loads back unchanged."""
        ckpt_path = self.root / "model_checkpoint.pth"
        payload = {
            "epoch": 5,
            "iteration": 150,
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0, 3.0])},
            "best_score": 0.88,
        }

        # Save atomically
        success = safe_atomic_save(payload, ckpt_path)
        self.assertTrue(success)
        self.assertTrue(ckpt_path.exists())

        # Check no .tmp file left
        self.assertFalse((self.root / "model_checkpoint.pth.tmp").exists())

        # Load safely
        loaded = safe_load_checkpoint(ckpt_path)
        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["iteration"], 150)
        self.assertEqual(loaded["best_score"], 0.88)
        self.assertTrue(
            torch.equal(loaded["model_state_dict"]["weight"], torch.tensor([1.0, 2.0, 3.0]))
        )

        # Audit disk space returns valid headroom
        headroom = audit_disk_space(self.root)
        self.assertGreater(headroom, 0.0)

    def test_resume_state_parsing_and_scaling(self) -> None:
        """Verify resume-state parsing and epoch/iteration scaling from a checkpoint payload."""
        payload = {
            "epoch": 3,
            "iteration": 50,
            "model_state_dict": {"layer.weight": torch.zeros(2, 2)},
            "optimizer_state_dict": {"param_groups": []},
            "sota_achieved": True,
        }
        state = parse_resume_state(payload)
        self.assertIsInstance(state, ResumeState)
        self.assertEqual(state.epoch, 3)
        self.assertEqual(state.iteration, 50)
        self.assertTrue(state.sota_achieved)
        self.assertIsNotNone(state.optimizer_state_dict)

        # Scale resume progress when loader size shrinks or expands
        scaled_half = scale_resume_progress(
            resume_iteration=50, source_loader_len=100, new_loader_len=50
        )
        self.assertEqual(scaled_half, 25)

        scaled_double = scale_resume_progress(
            resume_iteration=25, source_loader_len=50, new_loader_len=100
        )
        self.assertEqual(scaled_double, 50)

    def test_stretch_scheduler_runway(self) -> None:
        """Verify the scheduler runway is stretched when training is extended."""
        dummy_model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(dummy_model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

        sched_state = {
            "total_steps": 100,
            "last_epoch": 50,
            "_schedule_phases": [{"end_step": 30}, {"end_step": 99}],
        }
        stretch_scheduler_runway(sched, sched_state, current_total_steps=200)

        # Check total_steps was scaled
        self.assertEqual(sched_state["total_steps"], 200)
        self.assertEqual(sched_state["last_epoch"], 100)

    def test_checkpoint_recovery_engine(self) -> None:
        """Verify the recovery engine discovers checkpoints under canonical roots."""
        engine = CheckpointRecoveryEngine(
            workspace_root=self.root, project_root=self.root, env="local"
        )

        # Create mock checkpoint tree
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_epoch_001.pth").touch()
        (ckpt_dir / "model_epoch_002.pth").touch()
        (ckpt_dir / "model_best.pth").touch()
        (ckpt_dir / "model_latest.pth").touch()

        roots = engine.find_candidate_roots("my_model")
        self.assertIn(ckpt_dir, roots)

        discovered = engine.discover_checkpoints(roots, "my_model")
        self.assertIn("latest", discovered)
        self.assertIn("best", discovered)
        self.assertIn("latest_epoch", discovered)
        self.assertEqual(discovered["latest_epoch"].name, "model_epoch_002.pth")

        # Sync checkpoints
        target_dir = self.root / "target_hub"
        synced = engine.sync_to_local_hub(ckpt_dir, target_dir, "my_model")
        self.assertGreaterEqual(len(synced), 4)

    def test_metric_vault(self) -> None:
        """Verify the metric vault tracks the best value per metric and direction."""
        vault = MetricVault(primary_metric="val_loss", mode="min")

        # Epoch 1
        is_best1 = vault.record_epoch(1, {"val_loss": 0.50, "val_psnr": 28.0})
        self.assertTrue(is_best1)

        # Epoch 2 (improved)
        is_best2 = vault.record_epoch(2, {"val_loss": 0.40, "val_psnr": 29.5})
        self.assertTrue(is_best2)

        # Epoch 3 (worse)
        is_best3 = vault.record_epoch(3, {"val_loss": 0.45, "val_psnr": 29.0})
        self.assertFalse(is_best3)

        self.assertIsNotNone(vault.best_record)
        assert vault.best_record is not None
        self.assertEqual(vault.best_record.epoch, 2)
        self.assertAlmostEqual(vault.best_record.primary_value, 0.40)

        # Rolling average
        avg_loss = vault.get_rolling_average("val_loss", window=2)
        self.assertIsNotNone(avg_loss)
        assert avg_loss is not None
        self.assertAlmostEqual(avg_loss, 0.425)

        # Export and re-load CSV
        csv_path = self.root / "metrics.csv"
        vault.export_csv(csv_path)
        self.assertTrue(csv_path.exists())

        loaded_vault = MetricVault(primary_metric="val_loss", mode="min")
        loaded_vault.load_csv(csv_path)
        self.assertEqual(len(loaded_vault.history), 3)
        self.assertIsNotNone(loaded_vault.best_record)
        assert loaded_vault.best_record is not None
        self.assertEqual(loaded_vault.best_record.epoch, 2)

    def test_checkpoint_lifecycle_manager(self) -> None:
        """Verify the checkpoint lifecycle manager saves, rotates and resolves checkpoints."""
        mgr = CheckpointLifecycleManager(
            model_name="test_model",
            project_root=self.isolated_project_root(),
            sota_targets={"psnr": 30.0, "ssim": 0.90},
            min_progress_interval_sec=0.0,  # Zero for deterministic test evaluation
        )

        dummy_payload = {"epoch": 1, "state": "test"}

        # 1. Progress bounds test: < 2% rejected
        saved_low = mgr.check_and_save_progress(
            epoch=1,
            phase="train",
            step=1,
            total_steps=100,
            payload_builder=lambda: dict(dummy_payload),
        )
        self.assertFalse(saved_low)
        self.assertFalse(mgr.progress_path.exists())

        # 2. Progress bounds test: > 98% rejected
        saved_high = mgr.check_and_save_progress(
            epoch=1,
            phase="train",
            step=99,
            total_steps=100,
            payload_builder=lambda: dict(dummy_payload),
        )
        self.assertFalse(saved_high)
        self.assertFalse(mgr.progress_path.exists())

        # 3. Progress bounds test: within [2%, 98%] (e.g. 25%) accepted
        saved_valid = mgr.check_and_save_progress(
            epoch=1,
            phase="train",
            step=25,
            total_steps=100,
            payload_builder=lambda: dict(dummy_payload),
        )
        self.assertTrue(saved_valid)
        self.assertTrue(mgr.progress_path.exists())

        # 4. Save latest purges progress
        mgr.save_latest(epoch=1, payload=dict(dummy_payload))
        self.assertTrue(mgr.latest_path.exists())
        self.assertFalse(mgr.progress_path.exists())

        # 5. Save best
        is_best = mgr.save_best(epoch=1, quality_score=85.0, payload=dict(dummy_payload))
        self.assertTrue(is_best)
        self.assertTrue(mgr.best_path.exists())

        # Worse quality score rejected
        is_not_best = mgr.save_best(epoch=2, quality_score=80.0, payload=dict(dummy_payload))
        self.assertFalse(is_not_best)

        # 6. Vault milestones
        vaults_ep1 = mgr.check_and_save_vault(
            epoch=1,
            metrics={"psnr": 28.5, "ssim": 0.88},
            payload=dict(dummy_payload),
        )
        self.assertIn("psnr", vaults_ep1)
        self.assertIn("ssim", vaults_ep1)
        self.assertTrue((mgr.checkpoints_dir / "vault_psnr.pth").exists())
        self.assertTrue((mgr.checkpoints_dir / "vault_ssim.pth").exists())

        # Epoch 2 beats psnr but not ssim
        vaults_ep2 = mgr.check_and_save_vault(
            epoch=2,
            metrics={"psnr": 29.5, "ssim": 0.85},
            payload=dict(dummy_payload),
        )
        self.assertIn("psnr", vaults_ep2)
        self.assertNotIn("ssim", vaults_ep2)

    def test_vault_direction_and_alias_matching(self) -> None:
        """Verify vault metric directions and alias matching for forex metrics."""
        mgr = CheckpointLifecycleManager(
            model_name="forex_test",
            project_root=self.isolated_project_root(),
            sota_targets={
                "dir_acc": 58.5,
                "max_drawdown": 12.0,
                "sharpe_ratio": 1.85,
                "tp_mae": 12.5,
            },
            min_progress_interval_sec=0.0,
        )
        dummy_payload = {"epoch": 1}

        # Epoch 1: Telemetry dictionary with CSV/engine keys ("DirAcc", "MaxDD", "Sharpe", "TP_MAE")
        vaults_ep1 = mgr.check_and_save_vault(
            epoch=1,
            metrics={"DirAcc": 59.0, "MaxDD": 11.5, "Sharpe": 1.90, "TP_MAE": 12.0},
            payload=dummy_payload,
        )
        self.assertIn("dir_acc", vaults_ep1)
        self.assertIn("max_drawdown", vaults_ep1)
        self.assertIn("sharpe_ratio", vaults_ep1)
        self.assertIn("tp_mae", vaults_ep1)
        self.assertTrue((mgr.checkpoints_dir / "vault_dir_acc.pth").exists())
        self.assertTrue((mgr.checkpoints_dir / "vault_max_drawdown.pth").exists())
        self.assertTrue((mgr.checkpoints_dir / "vault_sharpe_ratio.pth").exists())
        self.assertTrue((mgr.checkpoints_dir / "vault_tp_mae.pth").exists())

        # Epoch 2: Only MaxDD improves (10.0 < 11.5). DirAcc, Sharpe are worse.
        # TP_MAE is worse (13.0 > 12.0).
        vaults_ep2 = mgr.check_and_save_vault(
            epoch=2,
            metrics={"DirAcc": 58.0, "MaxDD": 10.0, "Sharpe": 1.70, "TP_MAE": 13.0},
            payload=dummy_payload,
        )
        self.assertIn("max_drawdown", vaults_ep2)
        self.assertNotIn("dir_acc", vaults_ep2)
        self.assertNotIn("sharpe_ratio", vaults_ep2)
        self.assertNotIn("tp_mae", vaults_ep2)

    def test_resume_state_validation_phase(self) -> None:
        """Verify resume-state validation reads the progress-metadata phase."""
        payload = {
            "epoch": 2,
            "progress_metadata": {
                "epoch": 2,
                "phase": "val",
                "step": 13410,
                "total_steps": 27451,
            },
            "train_metrics": {"train_loss": 0.2714},
            "model_state": {"w": torch.tensor([1.0])},
        }
        state = parse_resume_state(payload)
        self.assertTrue(state.is_intra_epoch)
        self.assertEqual(state.phase, "val")
        self.assertEqual(state.epoch, 2)
        self.assertEqual(state.iteration, 13410)
        self.assertEqual(state.train_metrics["train_loss"], 0.2714)

    def test_checkpoint_format_version_and_migration(self) -> None:
        """Verify format version attachment and migration error handling."""
        # 1. Normal save attaches current format version
        ckpt_path = self.root / "v2_checkpoint.pth"
        payload = {"epoch": 1, "model_state_dict": {"w": torch.tensor([1.0])}}
        safe_atomic_save(payload, ckpt_path)
        loaded = safe_load_checkpoint(ckpt_path)
        self.assertEqual(loaded.get("checkpoint_format_version"), CHECKPOINT_FORMAT_VERSION)

        # 2. Legacy checkpoint without version migrates legacy state_dict key
        legacy_path = self.root / "legacy_checkpoint.pth"
        legacy_payload = {"epoch": 0, "state_dict": {"bias": torch.tensor([0.5])}}
        torch.save(legacy_payload, str(legacy_path))
        loaded_legacy = safe_load_checkpoint(legacy_path)
        self.assertIn("model_state_dict", loaded_legacy)
        self.assertTrue(torch.equal(loaded_legacy["model_state_dict"]["bias"], torch.tensor([0.5])))

        # 3. Future format version raises explicit CheckpointMigrationError
        future_path = self.root / "future_checkpoint.pth"
        future_payload = {"epoch": 10, "checkpoint_format_version": 999}
        torch.save(future_payload, str(future_path))
        with self.assertRaises(CheckpointMigrationError):
            safe_load_checkpoint(future_path)

    def test_bit_reproducible_rng_and_loss_continuity(self) -> None:
        """Verify kill-and-resume continuity: saving RNG + optim reproduces exact next steps."""
        torch.manual_seed(1337)
        model = torch.nn.Linear(4, 1, bias=False)
        model.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x_step1 = torch.randn(2, 4)
        x_step2 = torch.randn(2, 4)

        # Step 1
        loss1 = (model(x_step1) ** 2).mean()
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Capture checkpoint at Step 1 (simulating interrupt / kill)
        ckpt_path = self.root / "kill_and_resume.pth"
        payload = {
            "epoch": 1,
            "step": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **rng_payload(),
        }
        safe_atomic_save(payload, ckpt_path)

        # Reference: continue running Step 2 uninterrupted
        loss2_ref = (model(x_step2) ** 2).mean()
        loss2_ref.backward()
        optimizer.step()
        optimizer.zero_grad()
        ref_weight = model.weight.data.clone()

        # Resume simulation: fresh model and optimizer loaded from checkpoint
        resumed_model = torch.nn.Linear(4, 1, bias=False)
        resumed_optim = torch.optim.SGD(resumed_model.parameters(), lr=0.1)

        loaded_ckpt = safe_load_checkpoint(ckpt_path)
        resumed_state = parse_resume_state(loaded_ckpt)
        resumed_model.load_state_dict(resumed_state.model_state_dict)
        if resumed_state.optimizer_state_dict:
            resumed_optim.load_state_dict(resumed_state.optimizer_state_dict)
        restore_rng_state(loaded_ckpt)

        # Step 2 on resumed model
        loss2_resumed = (resumed_model(x_step2) ** 2).mean()
        loss2_resumed.backward()
        resumed_optim.step()
        resumed_optim.zero_grad()

        # Assert loss continuity and exact bit-reproducibility
        self.assertAlmostEqual(loss2_ref.item(), loss2_resumed.item(), places=6)
        self.assertTrue(torch.equal(ref_weight, resumed_model.weight.data))

    def test_vault_consistency_validator_and_pruning(self) -> None:
        """Verify vault consistency auditing, corruption detection, and garbage collection."""
        ckpt_dir = self.root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Valid checkpoints: best.pth, latest.pth, vault_psnr.pth
        best_p = ckpt_dir / "best.pth"
        latest_p = ckpt_dir / "latest.pth"
        vault_psnr_p = ckpt_dir / "vault_psnr.pth"
        payload = {"epoch": 2, "model_state": {"w": torch.tensor([1.0])}}
        safe_atomic_save(payload, best_p)
        safe_atomic_save(payload, latest_p)
        safe_atomic_save(payload, vault_psnr_p)

        report = validate_vault_consistency("test_model", ckpt_dir, {"psnr": 30.0})
        self.assertTrue(report.is_valid)
        self.assertIn("vault_psnr.pth", report.checkpoints_found)
        self.assertEqual(len(report.missing_targets), 0)

        # 2. Add stale progress.pth (epoch 1 vs latest epoch 2) and orphan temp file
        progress_p = ckpt_dir / "progress.pth"
        safe_atomic_save({"epoch": 1, "model_state": {"w": torch.tensor([1.0])}}, progress_p)
        orphan_tmp = ckpt_dir / "stray.tmp"
        orphan_tmp.write_text("incomplete write", encoding="utf-8")

        stale_report = validate_vault_consistency("test_model", ckpt_dir)
        self.assertFalse(stale_report.is_valid)
        self.assertIn("progress.pth", stale_report.stale_checkpoints)

        # 3. Prune vault: removes stray .tmp and stale progress.pth
        pruned = prune_vault_checkpoints(ckpt_dir)
        self.assertIn("stray.tmp", pruned)
        self.assertIn("progress.pth", pruned)
        self.assertFalse(orphan_tmp.exists())
        self.assertFalse(progress_p.exists())

        # 4. Corruption detection: empty (0-byte) checkpoint file
        corrupt_p = ckpt_dir / "corrupted.pth"
        corrupt_p.touch()
        corrupt_report = validate_vault_consistency("test_model", ckpt_dir)
        self.assertFalse(corrupt_report.is_valid)
        self.assertIn("corrupted.pth", corrupt_report.corrupted_checkpoints)

    def _stage_remote_vault(self) -> tuple[Path, list[torch.nn.Parameter]]:
        source_dir = self.root / "source_hub" / "model_x" / "checkpoints"
        source_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 2)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        save_payload = {
            "epoch": 5,
            "step": 500,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            **rng_payload(),
        }
        safe_atomic_save(save_payload, source_dir / "vault_psnr.pth")
        safe_atomic_save(save_payload, source_dir / "latest.pth")
        remote_store = self.root / "remote_cloud_storage"
        remote_store.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, remote_store / "checkpoints")
        return remote_store, list(model.parameters())

    def test_e2e_vault_save_download_fresh_container_resume(self) -> None:
        """Simulate end-to-end: training save -> upload staging -> download to fresh -> resume."""
        remote_store, orig_params = self._stage_remote_vault()
        fresh_container_root = self.root / "fresh_container"
        dest_checkpoints = fresh_container_root / "LemGendaryModels" / "model_x" / "checkpoints"
        recovery_engine = CheckpointRecoveryEngine(
            workspace_root=fresh_container_root,
            project_root=fresh_container_root / "suite",
            env="local",
        )
        recovery_engine.sync_to_local_hub(
            source_root=remote_store / "checkpoints",
            target_dir=dest_checkpoints,
            model_name="model_x",
        )
        self.assertTrue((dest_checkpoints / "vault_psnr.pth").exists())
        self.assertTrue((dest_checkpoints / "latest.pth").exists())

        report = validate_vault_consistency("model_x", dest_checkpoints, {"psnr": 30.0})
        self.assertTrue(report.is_valid)

        resumed_model = torch.nn.Linear(2, 2)
        resumed_optim = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
        resumed_data = safe_load_checkpoint(dest_checkpoints / "vault_psnr.pth")
        resume_state = parse_resume_state(resumed_data)
        resumed_model.load_state_dict(resume_state.model_state_dict)
        if resume_state.optimizer_state_dict:
            resumed_optim.load_state_dict(resume_state.optimizer_state_dict)
        restore_rng_state(resumed_data)

        self.assertEqual(resume_state.epoch, 5)
        for p1, p2 in zip(orig_params, resumed_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_resume_iteration_scales_across_batch_sizes(self) -> None:
        """Verify resume_iteration scales proportionally when loader length changes."""
        payload = {
            "epoch": 42,
            "phase": "train",
            "progress_metadata": {
                "step": 2243,
                "total_steps": 3887,
                "phase": "train",
            },
            "model_state": {"w": torch.tensor([1.0])},
        }
        state = parse_resume_state(payload)
        self.assertEqual(state.total_steps, 3887)
        self.assertEqual(state.iteration, 2243)

        # Same loader size returns exact iteration
        same_step = resume_iteration(state, "train", False, current_loader_len=3887)
        self.assertEqual(same_step, 2243)

        # Scaled loader size (batch size shrunk 256 -> 32; loader expanded 3887 -> 31097)
        scaled_step = resume_iteration(state, "train", False, current_loader_len=31097)
        # Expected: int((2243 / 3887) * 31097) = 17946
        expected = int((2243 / 3887.0) * 31097)
        self.assertEqual(scaled_step, expected)
        self.assertAlmostEqual(scaled_step / 31097.0, 2243 / 3887.0, places=4)


if __name__ == "__main__":
    unittest.main()
