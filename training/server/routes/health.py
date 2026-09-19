"""Health and operational telemetry endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from fastapi import APIRouter
import torch

from training.hardware.discovery import discover_device

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health() -> dict[str, Any]:
    """Return health status, daemon version, and hardware accelerator profile."""
    device_info = discover_device()
    return {
        "status": "ok",
        "service": "lemgendary-training-suite",
        "version": "2026.11.0",
        "port": 8200,
        "timestamp": datetime.now().isoformat(),
        "device": {
            "name": device_info.device_names[0] if device_info.device_names else str(device_info.device),
            "type": device_info.device_type,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": device_info.device_count if device_info.is_cuda else 0,
        },
    }
