"""Unit tests for governance subsystem: metrics, curriculum, thermal, sota, and governor."""

import unittest
from training.governance.curriculum import CurriculumState
from training.governance.governor import GovernorStateError, SmartTrainingGovernor
from training.governance.metrics import MetricRegistry
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState
from training.optimization_engine import (
    CurriculumState as LegacyCurriculumState,
    SmartTrainingGovernor as LegacyGovernor,
)


class TestMetricRegistry(unittest.TestCase):
    """Tests for MetricRegistry directionality, weights, and scoring."""

    def test_default_directions(self) -> None:
        reg = MetricRegistry()
        self.assertFalse(reg.is_higher_better("val_loss"))
        self.assertFalse(reg.is_higher_better("lpips"))
        self.assertFalse(reg.is_higher_better("fid"))
        self.assertTrue(reg.is_higher_better("psnr"))
        self.assertTrue(reg.is_higher_better("ssim"))
        self.assertTrue(reg.is_higher_better("srcc"))
        self.assertTrue(reg.is_higher_better("win_rate"))

    def test_is_improved(self) -> None:
        reg = MetricRegistry()
        # Loss: lower is better
        self.assertTrue(reg.is_improved("val_loss", new_val=0.45, baseline_val=0.50))
        self.assertFalse(reg.is_improved("val_loss", new_val=0.55, baseline_val=0.50))
        # PSNR: higher is better
        self.assertTrue(reg.is_improved("psnr", new_val=28.5, baseline_val=28.0))
        self.assertFalse(reg.is_improved("psnr", new_val=27.5, baseline_val=28.0))

    def test_composite_score_vision(self) -> None:
        reg = MetricRegistry()
        metrics = {"psnr": 30.0, "ssim": 0.92, "lpips": 0.12}
        score = reg.compute_composite_score(metrics, task_type="quality")
        self.assertGreater(score, 0.0)

    def test_composite_score_forex(self) -> None:
        reg = MetricRegistry()
        forex_metrics = {
            "dir_acc": 0.65,
            "win_rate": 0.58,
            "profit_factor": 1.75,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.08,
        }
        score = reg.compute_composite_score(forex_metrics, task_type="forex")
        self.assertGreater(score, 0.0)


class TestCurriculumState(unittest.TestCase):
    """Tests for CurriculumState resolution and batch management."""

    def test_resolution_promotion(self) -> None:
        curr = CurriculumState(res_ladder=[256, 384, 512], current_res=256)
        self.assertTrue(curr.promote_resolution())
        self.assertEqual(curr.current_res, 384)
        self.assertTrue(curr.promote_resolution())
        self.assertEqual(curr.current_res, 512)
        self.assertFalse(curr.promote_resolution())
        self.assertEqual(curr.current_res, 512)

    def test_fraction_expansion(self) -> None:
        curr = CurriculumState(current_fraction=0.15, fraction_increment=0.15)
        self.assertTrue(curr.expand_fraction())
        self.assertAlmostEqual(curr.current_fraction, 0.30, places=2)
        curr.current_fraction = 1.0
        self.assertFalse(curr.expand_fraction())

    def test_batch_and_accumulation(self) -> None:
        curr = CurriculumState(target_effective_batch=32, current_batch=8)
        curr.update_batch_and_accumulation(16)
        self.assertEqual(curr.current_batch, 16)
        self.assertEqual(curr.current_acc, 2)

    def test_serialization(self) -> None:
        curr = CurriculumState(res_ladder=[256, 512], current_res=512, current_fraction=0.8)
        data = curr.to_dict()
        restored = CurriculumState.from_dict(data)
        self.assertEqual(restored.res_ladder, [256, 512])
        self.assertEqual(restored.current_res, 512)
        self.assertAlmostEqual(restored.current_fraction, 0.8)


class TestThermalState(unittest.TestCase):
    """Tests for ThermalState temperature decay and turbulence dampening."""

    def test_step_epoch(self) -> None:
        thermal = ThermalState(temperature=1.0, cooling_factor=0.9, min_temperature=0.2)
        new_temp = thermal.step_epoch()
        self.assertAlmostEqual(new_temp, 0.9, places=2)

    def test_dampen_turbulence(self) -> None:
        thermal = ThermalState(temperature=0.5, turbulence_dampening=True)
        boosted = thermal.dampen_turbulence(gradient_norm=15.0, threshold=10.0)
        self.assertTrue(boosted)
        self.assertGreater(thermal.temperature, 0.5)

    def test_serialization(self) -> None:
        thermal = ThermalState(temperature=0.75, cooling_factor=0.95)
        data = thermal.to_dict()
        restored = ThermalState.from_dict(data)
        self.assertEqual(restored.temperature, 0.75)
        self.assertEqual(restored.cooling_factor, 0.95)


class TestSotaTracker(unittest.TestCase):
    """Tests for SotaTracker metric recording and rollback logic."""

    def test_record_epoch_new_sota(self) -> None:
        tracker = SotaTracker()
        is_sota = tracker.record_epoch(current_quality=0.82)
        self.assertTrue(is_sota)
        self.assertEqual(tracker.best_quality, 0.82)
        self.assertEqual(len(tracker.history), 1)

    def test_reset_best(self) -> None:
        tracker = SotaTracker(best_quality=0.95, prev_quality=0.94)
        tracker.reset_best()
        self.assertEqual(tracker.best_quality, 0.0)
        self.assertEqual(tracker.stabilization_epochs, 2)
        self.assertEqual(len(tracker.history), 0)

    def test_register_rollback_and_breakout(self) -> None:
        tracker = SotaTracker(
            loop_breaker_enabled=True,
            loop_breaker_threshold=2,
            loop_breaker_strategy="escalate",
            task_type="quality",
        )
        res_ladder = [256, 384, 512]
        res1 = tracker.register_rollback(current_res=256, res_ladder=res_ladder)
        self.assertFalse(res1["breakout_triggered"])

        res2 = tracker.register_rollback(current_res=256, res_ladder=res_ladder)
        self.assertTrue(res2["breakout_triggered"])
        self.assertEqual(res2["new_res"], 384)


class TestSmartTrainingGovernor(unittest.TestCase):
    """Tests for complete SmartTrainingGovernor coordinator."""

    def setUp(self) -> None:
        self.model_info = {
            "name": "test_governor_model",
            "dataset_type": "quality",
            "input_size": 256,
            "batch_size": 8,
            "optimization": {
                "res_ladder": [256, 384, 512],
                "initial_fraction": 0.20,
                "target_effective_batch": 16,
            },
        }

    def test_initialization_and_properties(self) -> None:
        gov = SmartTrainingGovernor(self.model_info)
        self.assertEqual(gov.current_res, 256)
        self.assertAlmostEqual(gov.current_fraction, 0.20)
        self.assertEqual(gov.current_batch, 8)
        self.assertEqual(gov.res_ladder, [256, 384, 512])

        # Test setters
        gov.current_fraction = 0.50
        self.assertEqual(gov.current_fraction, 0.50)
        gov.current_res = 384
        self.assertEqual(gov.current_res, 384)

    def test_suggest_batch_growth(self) -> None:
        gov = SmartTrainingGovernor(self.model_info)
        new_b, new_acc = gov.suggest_batch_growth(
            current_batch=8,
            current_acc=2,
            target_eff=16,
            vram_free_ratio=0.55,
        )
        self.assertEqual(new_b, 16)
        self.assertEqual(new_acc, 1)

    def test_audit_epoch_returns_8_tuple(self) -> None:
        gov = SmartTrainingGovernor(self.model_info)
        res = gov.audit_epoch(
            current_quality=0.85,
            best_quality=0.80,
            epochs_no_improve=0,
            regression_epochs=0,
        )
        self.assertEqual(len(res), 8)
        f_chg, r_chg, lr_chg, t_chg, c_chg, b_chg, stop_trig, msg = res
        self.assertIsInstance(stop_trig, bool)
        self.assertIsInstance(msg, str)

    def test_get_and_load_state_roundtrip(self) -> None:
        gov = SmartTrainingGovernor(self.model_info)
        gov.current_fraction = 0.65
        gov.current_res = 384
        gov.best_quality = 0.91

        state = gov.get_state()
        self.assertEqual(state["sample_fraction"], 0.65)
        self.assertEqual(state["input_size"], 384)
        self.assertEqual(state["best_quality"], 0.91)

        new_gov = SmartTrainingGovernor(self.model_info)
        new_gov.load_state(state)
        self.assertAlmostEqual(new_gov.current_fraction, 0.65)
        self.assertEqual(new_gov.current_res, 384)
        self.assertEqual(new_gov.best_quality, 0.91)

    def test_legacy_shim_compatibility(self) -> None:
        gov = LegacyGovernor(self.model_info)
        self.assertIsInstance(gov, SmartTrainingGovernor)
        self.assertEqual(gov.current_res, 256)


class TestYOLOLadderCurriculum(unittest.TestCase):
    """Unit tests for YOLO ladder curriculum and gradual 15-20% intra-resolution fraction progression."""

    def test_ladder_intra_resolution_fractions(self) -> None:
        from training.governance.yolo_governor import build_ladder_curriculum
        opt_cfg = {"initial_fraction": 0.3, "plateau_patience": 10}
        stages = build_ladder_curriculum(
            res_ladder=[320, 480, 640],
            total_epochs=300,
            vram_gb=4.0,
            requested_batch=16,
            opt_config=opt_cfg,
        )
        # 13 stages total: 5 on 320px, 4 on 480px, 4 on 640px
        self.assertEqual(len(stages), 13)

        # Base rung (320px) traverses 30% -> 50% -> 70% -> 85% -> 100%
        base_fracs = [stages[i].fraction for i in range(5)]
        self.assertEqual(base_fracs, [0.30, 0.50, 0.70, 0.85, 1.00])
        for i in range(4):
            delta = round(base_fracs[i + 1] - base_fracs[i], 2)
            self.assertTrue(0.15 <= delta <= 0.20, f"Delta {delta} not in [0.15, 0.20]")

        # 480px rung starts at 50% and progresses: 50% -> 65% -> 80% -> 100%
        rung480_fracs = [stages[i].fraction for i in range(5, 9)]
        self.assertEqual(rung480_fracs, [0.50, 0.65, 0.80, 1.00])
        for i in range(3):
            delta = round(rung480_fracs[i + 1] - rung480_fracs[i], 2)
            self.assertTrue(0.15 <= delta <= 0.20, f"Delta {delta} not in [0.15, 0.20]")

        # 640px rung starts at 50% and progresses: 50% -> 65% -> 80% -> 100%
        rung640_fracs = [stages[i].fraction for i in range(9, 13)]
        self.assertEqual(rung640_fracs, [0.50, 0.65, 0.80, 1.00])
        for i in range(3):
            delta = round(rung640_fracs[i + 1] - rung640_fracs[i], 2)
            self.assertTrue(0.15 <= delta <= 0.20, f"Delta {delta} not in [0.15, 0.20]")

        # Epoch sum equality
        self.assertEqual(sum(s.target_epochs for s in stages), 300)

    def test_ladder_start_resolution_override(self) -> None:
        from training.governance.yolo_governor import build_ladder_curriculum
        stages = build_ladder_curriculum(
            res_ladder=[320, 480, 640],
            total_epochs=100,
            vram_gb=8.0,
            requested_batch=16,
            opt_config={},
            start_resolution=480,
        )
        # 8 stages total: 4 on 480px, 4 on 640px
        self.assertEqual(len(stages), 8)
        self.assertEqual(stages[0].resolution, 480)
        self.assertEqual(stages[0].fraction, 0.50)
        self.assertEqual(stages[3].resolution, 480)
        self.assertEqual(stages[3].fraction, 1.00)
        self.assertEqual(stages[4].resolution, 640)
        self.assertEqual(stages[4].fraction, 0.50)
        self.assertEqual(stages[7].resolution, 640)
        self.assertEqual(stages[7].fraction, 1.00)
        self.assertEqual(sum(s.target_epochs for s in stages), 100)

    def test_ladder_single_resolution(self) -> None:
        from training.governance.yolo_governor import build_ladder_curriculum
        stages = build_ladder_curriculum(
            res_ladder=[640],
            total_epochs=50,
            vram_gb=8.0,
            requested_batch=8,
            opt_config={},
        )
        # 4 stages total: 50% -> 65% -> 80% -> 100%
        self.assertEqual(len(stages), 4)
        self.assertEqual([s.fraction for s in stages], [0.50, 0.65, 0.80, 1.00])
        self.assertEqual(sum(s.target_epochs for s in stages), 50)



class TestYOLOCheckpointIsolationAndSotaExport(unittest.TestCase):
    """Tests to verify checkpoint isolation to checkpoints subfolders and SOTA export behavior."""

    def test_checkpoint_isolation_directories(self) -> None:
        import argparse
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock
        from training.governance.yolo_governor import YOLOCurriculumGovernor

        with tempfile.TemporaryDirectory() as tmpdir:
            proj_root = Path(tmpdir) / "lemgendary-training-suite"
            models_root = Path(tmpdir) / "LemGendaryModels" / "yolov8n"
            proj_root.mkdir(parents=True)
            models_root.mkdir(parents=True)

            mock_args = argparse.Namespace(
                clean=False,
                start_res=None,
                target_epochs=100,
                auto_sync=False,
                env="local",
            )
            config = {
                "curriculum": {"resolutions": [320, 480, 640]},
                "training": {"batch_size": 16, "epochs": 100},
                "optimization": {},
            }

            governor = YOLOCurriculumGovernor(
                args=mock_args,
                config=config,
                project_root=proj_root,
            )

            # Create mock stage weights
            stage_dir = proj_root / "runs" / "exp_stage1"
            weights_dir = stage_dir / "weights"
            weights_dir.mkdir(parents=True)
            fake_best = weights_dir / "best.pt"
            fake_last = weights_dir / "last.pt"
            fake_best.write_bytes(b"FAKEMODELBEST")
            fake_last.write_bytes(b"FAKEMODELLAST")

            # Call checkpoint sync
            governor._synchronize_checkpoints(
                stage_dir=stage_dir,
                resolution=320,
                fraction=0.30,
                stage_index=1,
                global_epoch=1,
            )

            # Assert checkpoints exist strictly in checkpoints directories
            hub_ckpt_dir = governor.models_hub_ckpt_dir
            self.assertTrue((hub_ckpt_dir / "best.pt").exists())
            self.assertTrue((hub_ckpt_dir / "best.pth").exists())
            self.assertTrue((hub_ckpt_dir / "last.pt").exists())
            self.assertTrue((hub_ckpt_dir / "progress.pth").exists())
            self.assertTrue((hub_ckpt_dir / "curriculum_state.json").exists())

            # Assert root LemGendaryModels/yolov8n contains NO checkpoint artifacts
            self.assertFalse((governor.models_hub_dir / "best.pt").exists())
            self.assertFalse((governor.models_hub_dir / "best.pth").exists())
            self.assertFalse((governor.models_hub_dir / "last.pt").exists())
            self.assertFalse((governor.models_hub_dir / "progress.pth").exists())
            self.assertFalse((governor.models_hub_dir / "curriculum_state.json").exists())

    def test_sota_model_export(self) -> None:
        import argparse
        import tempfile
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from training.governance.yolo_governor import YOLOCurriculumGovernor

        with tempfile.TemporaryDirectory() as tmpdir:
            proj_root = Path(tmpdir) / "lemgendary-training-suite"
            models_root = Path(tmpdir) / "LemGendaryModels" / "yolov8n"
            proj_root.mkdir(parents=True)
            models_root.mkdir(parents=True)

            mock_args = argparse.Namespace(
                clean=False,
                start_res=None,
                target_epochs=100,
                auto_sync=False,
                env="local",
            )
            config = {
                "curriculum": {"resolutions": [320, 480, 640]},
                "training": {"batch_size": 16, "epochs": 100},
                "optimization": {},
            }

            governor = YOLOCurriculumGovernor(
                args=mock_args,
                config=config,
                project_root=proj_root,
            )

            fake_weight = proj_root / "best.pt"
            fake_weight.write_bytes(b"SOTAWEIGHTS")

            # Mock YOLO export so ONNX conversion produces a mock file
            mock_yolo_instance = MagicMock()
            fake_onnx_source = Path(tmpdir) / "mock_exported.onnx"
            fake_onnx_source.write_bytes(b"ONNXMODELDATA")
            mock_yolo_instance.export.return_value = str(fake_onnx_source)

            with patch("ultralytics.YOLO", return_value=mock_yolo_instance):
                governor._export_sota_models(fake_weight, resolution=640)

            # Verify PyTorch model and ONNX model were exported to LemGendaryModels/yolov8n
            exported_pt = governor.models_hub_dir / "yolov8n.pt"
            exported_onnx = governor.models_hub_dir / "yolov8n.onnx"
            self.assertTrue(exported_pt.exists())
            self.assertEqual(exported_pt.read_bytes(), b"SOTAWEIGHTS")
            self.assertTrue(exported_onnx.exists())
            self.assertEqual(exported_onnx.read_bytes(), b"ONNXMODELDATA")

    def test_open_ended_sota_convergence_protocol(self) -> None:
        import argparse
        import tempfile
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from training.governance.yolo_governor import YOLOCurriculumGovernor

        with tempfile.TemporaryDirectory() as tmpdir:
            proj_root = Path(tmpdir) / "lemgendary-training-suite"
            models_root = Path(tmpdir) / "LemGendaryModels" / "yolov8n"
            proj_root.mkdir(parents=True)
            models_root.mkdir(parents=True)

            mock_args = argparse.Namespace(
                clean=False,
                resolution=None,
                target_epochs=10,
                epochs=10,
                batch_size="16",
                auto_sync=False,
                env="local",
                lr=None,
            )
            config = {
                "curriculum": {"resolutions": [640]},
                "training": {"batch_size": 16, "epochs": 10},
                "optimization": {"res_ladder": [640], "extension_epochs": 5},
            }

            governor = YOLOCurriculumGovernor(
                args=mock_args,
                config=config,
                project_root=proj_root,
            )

            call_count = 0
            def mock_train(**kwargs):
                nonlocal call_count
                call_count += 1
                # When reaching extension cycle (call 5), simulate SOTA achievement
                if call_count >= 5:
                    governor.sota_achieved = True

            mock_yolo_instance = MagicMock()
            mock_yolo_instance.train.side_effect = mock_train
            fake_onnx = Path(tmpdir) / "test.onnx"
            fake_onnx.write_bytes(b"ONNXDATA")
            mock_yolo_instance.export.return_value = str(fake_onnx)

            with patch("ultralytics.YOLO", return_value=mock_yolo_instance), \
                 patch("training.governance.yolo_governor.generate_yolo_yaml", return_value="mock.yaml"):
                summary = governor.run()

            # Governor executed 4 planned stages (50%, 65%, 80%, 100%) + 1 extension cycle until SOTA
            self.assertEqual(call_count, 5)
            self.assertTrue(governor.sota_achieved)
            self.assertEqual(summary.status, "completed")


if __name__ == "__main__":
    unittest.main()



