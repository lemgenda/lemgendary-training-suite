"""Unit tests for the VRAM safety governor: hardware caps, ratchet guards and CPU fallback."""

import unittest
from operator import methodcaller

import torch

from tests.fixtures.models import SingleConvModel
from training.governance import SmartTrainingGovernor
from training.hardware.probe import VramProbe
from training.model_registry import audit_hardware_vram
from training.ratchet import RatchetViolation


class TestVRAMSafetyGovernor(unittest.TestCase):
    """Tests for VRAM-aware governor capping and resolution ratchet behaviour."""

    def test_governor_hardware_cap_restoration(self):
        """Verify the governor caps restoration batch/resolution to the hardware budget."""
        model_info = {
            "name": "test_nafnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {"res_ladder": [256, 384, 512, 640, 768]},
        }
        config = {"hardware": {"max_allowed_resolution": 384}}
        gov = SmartTrainingGovernor(model_info, config=config)
        # All rungs > 384 should be clamped
        self.assertTrue(all(r <= 384 for r in gov.res_ladder))
        self.assertIn(384, gov.res_ladder)
        self.assertNotIn(512, gov.res_ladder)
        self.assertNotIn(640, gov.res_ladder)
        self.assertNotIn(768, gov.res_ladder)

    def test_governor_commit_rung_never_retreats_resolution(self):
        """Verify committing a rung never moves the resolution backwards."""
        model_info = {
            "name": "test_mprnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {"res_ladder": [256, 384, 512]},
        }
        gov = SmartTrainingGovernor(model_info)
        # The governor's internal rung hook, called by name as the epoch audit does.
        methodcaller("_commit_rung", 512, 0.20)(gov)
        with self.assertRaises(RatchetViolation):
            methodcaller("_commit_rung", 384, 0.20)(gov)
        self.assertEqual(gov.current_res, 512)

    def test_audit_hardware_vram_cpu_fallback(self):
        """Verify the VRAM audit falls back to CPU-safe settings without CUDA."""
        model_info = {
            "name": "test_nafnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
        }
        config = {"defaults": {"batch_size": 8}}
        model = SingleConvModel()
        device = torch.device("cpu")
        batch = audit_hardware_vram(
            "nafnet_debluring", model_info, config, model, VramProbe(device, res_override=256)
        )
        self.assertEqual(batch, 8)


if __name__ == "__main__":
    unittest.main()
