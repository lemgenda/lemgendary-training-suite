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


if __name__ == "__main__":
    unittest.main()
