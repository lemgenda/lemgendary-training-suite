"""Unit tests for checkpoint management, recovery engine, resume scaling, and metric vault."""

from pathlib import Path
import tempfile
import unittest
import torch

from training.checkpoint import (
    CheckpointLifecycleManager,
    CheckpointRecoveryEngine,
    MetricVault,
    ResumeState,
    audit_disk_space,
    parse_resume_state,
    safe_atomic_save,
    safe_load_checkpoint,
    scale_resume_progress,
    stretch_scheduler_runway,
)


class TestCheckpointSubsystem(unittest.TestCase):
    """Test suite covering atomic checkpoint saves, recovery, resume scaling, and metric vault."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_atomic_save_and_load(self) -> None:
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
        self.assertTrue(torch.equal(loaded["model_state_dict"]["weight"], torch.tensor([1.0, 2.0, 3.0])))

        # Audit disk space returns valid headroom
        headroom = audit_disk_space(self.root)
        self.assertGreater(headroom, 0.0)

    def test_resume_state_parsing_and_scaling(self) -> None:
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
        scaled_half = scale_resume_progress(resume_iteration=50, source_loader_len=100, new_loader_len=50)
        self.assertEqual(scaled_half, 25)

        scaled_double = scale_resume_progress(resume_iteration=25, source_loader_len=50, new_loader_len=100)
        self.assertEqual(scaled_double, 50)

    def test_stretch_scheduler_runway(self) -> None:
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
        engine = CheckpointRecoveryEngine(workspace_root=self.root, project_root=self.root, env="local")

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

        self.assertEqual(vault.best_record.epoch, 2)
        self.assertAlmostEqual(vault.best_record.primary_value, 0.40)

        # Rolling average
        avg_loss = vault.get_rolling_average("val_loss", window=2)
        self.assertAlmostEqual(avg_loss, 0.425)

        # Export and re-load CSV
        csv_path = self.root / "metrics.csv"
        vault.export_csv(csv_path)
        self.assertTrue(csv_path.exists())

        loaded_vault = MetricVault(primary_metric="val_loss", mode="min")
        loaded_vault.load_csv(csv_path)
        self.assertEqual(len(loaded_vault.history), 3)
        self.assertEqual(loaded_vault.best_record.epoch, 2)

    def test_checkpoint_lifecycle_manager(self) -> None:
        mgr = CheckpointLifecycleManager(
            model_name="test_model",
            project_root=self.root,
            sota_targets={"psnr": 30.0, "ssim": 0.90},
            min_progress_interval_sec=0.0,  # Zero for deterministic test evaluation
        )

        dummy_payload = {"epoch": 1, "state": "test"}

        # 1. Progress bounds test: < 5% rejected
        saved_low = mgr.check_and_save_progress(
            epoch=1, phase="train", step=2, total_steps=100, payload_builder=lambda: dict(dummy_payload)
        )
        self.assertFalse(saved_low)
        self.assertFalse(mgr.progress_path.exists())

        # 2. Progress bounds test: > 50% rejected
        saved_high = mgr.check_and_save_progress(
            epoch=1, phase="train", step=60, total_steps=100, payload_builder=lambda: dict(dummy_payload)
        )
        self.assertFalse(saved_high)
        self.assertFalse(mgr.progress_path.exists())

        # 3. Progress bounds test: within [5%, 50%] (e.g. 25%) accepted
        saved_valid = mgr.check_and_save_progress(
            epoch=1, phase="train", step=25, total_steps=100, payload_builder=lambda: dict(dummy_payload)
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


if __name__ == "__main__":
    unittest.main()

