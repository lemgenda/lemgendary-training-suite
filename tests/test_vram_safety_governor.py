"""Unit tests for the VRAM safety governor: hardware caps, resolution vetoes and CPU fallback."""

import unittest

import torch

from tests.fixtures.models import SingleConvModel
from training.governance import SmartTrainingGovernor
from training.model_registry import audit_hardware_vram
from training.ratchet import RatchetViolation


class TestVRAMSafetyGovernor(unittest.TestCase):
    """Tests for VRAM-aware governor capping and resolution veto behaviour."""

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

    def test_governor_veto_resolution_jump(self):
        """Verify the governor vetoes a resolution jump that would exceed VRAM."""
        model_info = {
            "name": "test_mprnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {"res_ladder": [256, 384, 512]},
        }
        gov = SmartTrainingGovernor(model_info)
        gov.current_res = 384
        gov.veto_resolution_jump(384, reason="Test VRAM Dry-Run OOM")
        self.assertEqual(gov.current_res, 384)
        self.assertTrue(gov.spatial_lock_remaining > 0)
        self.assertTrue(gov.stabilization_epochs > 0)
        self.assertTrue(all(r <= 384 for r in gov.res_ladder))

    def test_governor_veto_never_retreats_resolution(self):
        """Verify a vetoed jump never moves the resolution backwards."""
        model_info = {
            "name": "test_mprnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {"res_ladder": [256, 384, 512]},
        }
        gov = SmartTrainingGovernor(model_info)
        gov.current_res = 512
        with self.assertRaises(RatchetViolation):
            gov.veto_resolution_jump(384, reason="Test VRAM Dry-Run OOM")
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
            "nafnet_debluring", model_info, config, device, model, res_override=256
        )
        self.assertEqual(batch, 8)


if __name__ == "__main__":
    unittest.main()
