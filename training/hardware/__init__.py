"""Hardware discovery, device policy, VRAM probing, and SentinelGuard for LemGendary Model Training Suite."""

from training.hardware.discovery import DeviceInfo, discover_device
from training.hardware.policy import HardwarePolicyResult, apply_hardware_policy
from training.hardware.probe import audit_hardware_vram
from training.hardware.sentinel import SentinelGuard

__all__ = [
    "DeviceInfo",
    "discover_device",
    "HardwarePolicyResult",
    "apply_hardware_policy",
    "audit_hardware_vram",
    "SentinelGuard",
]
