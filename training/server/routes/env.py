"""Environment and hardware telemetry endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from training.services.audit_service import AuditService

router = APIRouter(prefix="/env", tags=["env"])


@router.get("/telemetry")
def get_env_telemetry() -> dict[str, Any]:
    """Retrieve host system hardware telemetry, memory headroom, and disk capacity."""
    service = AuditService()
    return service.audit_system()
