"""Unit tests for training/hardware subsystem in LemGendary Model Training Suite."""

import unittest

import torch

from tests.fixtures.models import SingleConvModel
from training.hardware.discovery import DeviceInfo, discover_device
from training.hardware.policy import HardwarePolicyResult, apply_hardware_policy
from training.hardware.probe import audit_hardware_vram
from training.hardware.sentinel import SentinelGuard


class TestHardwareDiscovery(unittest.TestCase):
    """Tests device discovery and profile attributes."""

    def test_discover_device(self) -> None:
        """Verify device discovery returns a DeviceInfo with a torch device."""
        info = discover_device()
        self.assertIsInstance(info, DeviceInfo)
        self.assertIsInstance(info.device, torch.device)
        self.assertIsInstance(info.device_type, str)
        self.assertGreaterEqual(info.device_count, 1)
        self.assertGreaterEqual(info.total_vram_gb, 0.0)

    def test_discover_device_forced_cpu(self) -> None:
        """Verify forcing the CPU device is honoured."""
        info = discover_device(force_device="cpu")
        self.assertEqual(info.device.type, "cpu")
        self.assertTrue(info.is_cpu)
        self.assertFalse(info.is_cuda)


class TestHardwarePolicy(unittest.TestCase):
    """Tests execution policy application and AMP rules."""

    def test_policy_vision_model(self) -> None:
        """Verify the hardware policy for a vision model."""
        dev_info = discover_device(force_device="cpu")
        model_info = {"dataset_type": "restoration"}
        policy = apply_hardware_policy("mirnet_exposure", model_info, dev_info)
        self.assertIsInstance(policy, HardwarePolicyResult)
        # On CPU, AMP should be disabled
        self.assertFalse(policy.amp_enabled)

    def test_policy_forex_model_disables_amp(self) -> None:
        """Verify the hardware policy disables AMP for forex models."""
        dev_info = discover_device(force_device="cpu")
        model_info = {"dataset_type": "forex"}
        policy = apply_hardware_policy("forex_predictor", model_info, dev_info)
        # Forex models must always disable AMP to prevent FP16 overflow
        self.assertFalse(policy.amp_enabled)


class TestHardwareProbe(unittest.TestCase):
    """Tests VRAM probing and batch size calculations."""

    def test_audit_hardware_vram_cpu(self) -> None:
        """Verify the VRAM audit on CPU for a restoration model."""
        model = SingleConvModel()
        model_info = {"dataset_type": "restoration", "input_size": 224}
        config = {"defaults": {"batch_size": 8}}
        batch = audit_hardware_vram("test_mirnet", model_info, config, torch.device("cpu"), model)
        self.assertEqual(batch, 8)

    def test_audit_hardware_vram_forex(self) -> None:
        """Verify the VRAM audit for a forex model."""
        model = SingleConvModel()
        model_info = {"dataset_type": "forex", "batch_size": 128}
        config = {}
        batch = audit_hardware_vram(
            "forex_predictor", model_info, config, torch.device("cpu"), model
        )
        self.assertEqual(batch, 128)


class TestHardwareSentinel(unittest.TestCase):
    """Tests SentinelGuard memory checking, throttling, and veto logic."""

    def test_sentinel_cpu(self) -> None:
        """Verify the sentinel reports zero memory on CPU."""
        sentinel = SentinelGuard(torch.device("cpu"))
        mem = sentinel.check_memory()
        self.assertEqual(mem["total_gb"], 0.0)
        self.assertFalse(sentinel.is_memory_critical())

    def test_throttle_batch(self) -> None:
        """Verify batch throttling halves the batch but respects the minimum."""
        sentinel = SentinelGuard(torch.device("cpu"))
        throttled = sentinel.throttle_batch(64, min_batch=1)
        self.assertEqual(throttled, 32)
        min_throttled = sentinel.throttle_batch(1, min_batch=1)
        self.assertEqual(min_throttled, 1)

    def test_veto_resolution_jump_cpu(self) -> None:
        """Verify resolution jumps are permitted on CPU, with a reason string."""
        sentinel = SentinelGuard(torch.device("cpu"))
        vetoed, reason = sentinel.veto_resolution_jump(512, 256)
        self.assertFalse(vetoed)
        self.assertIsInstance(reason, str)

    def test_recover_memory(self) -> None:
        """Verify memory recovery runs on CPU without raising."""
        sentinel = SentinelGuard(torch.device("cpu"))
        # Should execute without raising
        sentinel.recover_memory()


if __name__ == "__main__":
    unittest.main()
