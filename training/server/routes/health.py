"""Health and operational telemetry endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import torch
from fastapi import APIRouter

from training.ecosystem import TRAINING_SERVICE, service_port
from training.hardware.discovery import discover_device
from training.version import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
@router.head("/health")
async def get_health() -> dict[str, Any]:
    """Return health status, daemon version, and hardware accelerator profile."""
    loop = asyncio.get_running_loop()
    device_info = await loop.run_in_executor(None, discover_device)
    return {
        "status": "ok",
        "service": "lemgendary-training-suite",
        "version": __version__,
        "port": service_port(TRAINING_SERVICE),
        "timestamp": datetime.now().isoformat(),
        "device": {
            "name": device_info.device_names[0]
            if device_info.device_names
            else str(device_info.device),
            "type": device_info.device_type,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": device_info.device_count if device_info.is_cuda else 0,
        },
    }
