"""Hardware discovery, device policy, VRAM probing, and SentinelGuard for LemGendary training."""

from training.hardware.discovery import DeviceInfo, discover_device
from training.hardware.policy import HardwarePolicyResult, apply_hardware_policy
from training.hardware.probe import VramProbe, audit_hardware_vram
from training.hardware.sentinel import SentinelGuard

__all__ = [
    "DeviceInfo",
    "HardwarePolicyResult",
    "SentinelGuard",
    "VramProbe",
    "apply_hardware_policy",
    "audit_hardware_vram",
    "discover_device",
]
