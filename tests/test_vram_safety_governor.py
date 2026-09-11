import unittest
import torch
import torch.nn as nn
from training.optimization_engine import SmartTrainingGovernor
from training.model_registry import audit_hardware_vram

class DummyRestorationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["pixel_values"]
        return self.conv(x)

class TestVRAMSafetyGovernor(unittest.TestCase):
    def test_governor_hardware_cap_restoration(self):
        model_info = {
            "name": "test_nafnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {
                "res_ladder": [256, 384, 512, 640, 768]
            }
        }
        config = {
            "hardware": {
                "max_allowed_resolution": 384
            }
        }
        gov = SmartTrainingGovernor(model_info, config=config)
        # All rungs > 384 should be clamped
        self.assertTrue(all(r <= 384 for r in gov.res_ladder))
        self.assertIn(384, gov.res_ladder)
        self.assertNotIn(512, gov.res_ladder)
        self.assertNotIn(640, gov.res_ladder)
        self.assertNotIn(768, gov.res_ladder)

    def test_governor_veto_resolution_jump(self):
        model_info = {
            "name": "test_mprnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto",
            "optimization": {
                "res_ladder": [256, 384, 512]
            }
        }
        gov = SmartTrainingGovernor(model_info)
        gov.current_res = 512
        gov.veto_resolution_jump(384, reason="Test VRAM Dry-Run OOM")
        self.assertEqual(gov.current_res, 384)
        self.assertTrue(gov.spatial_lock_remaining > 0)
        self.assertTrue(gov.stabilization_epochs > 0)
        self.assertTrue(all(r <= 384 for r in gov.res_ladder))

    def test_audit_hardware_vram_cpu_fallback(self):
        model_info = {
            "name": "test_nafnet",
            "dataset_type": "restoration",
            "input_size": 256,
            "batch_size": "auto"
        }
        config = {"defaults": {"batch_size": 8}}
        model = DummyRestorationModel()
        device = torch.device("cpu")
        batch = audit_hardware_vram("nafnet_debluring", model_info, config, device, model, res_override=256)
        self.assertEqual(batch, 8)

if __name__ == "__main__":
    unittest.main()
