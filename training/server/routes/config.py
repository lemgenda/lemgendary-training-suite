"""Configuration and preset discovery endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
import yaml

from training.services.training_service import TrainingService
from training.utils.paths import get_project_root

router = APIRouter(tags=["config"])


def _get_project_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    """Retrieve suite configuration manifest."""
    root = _get_project_root(request)
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@router.get("/presets")
def get_presets(request: Request) -> dict[str, Any]:
    """List all canonical training presets."""
    root = _get_project_root(request)
    service = TrainingService(project_root=root)
    return service.load_presets()


@router.get("/presets/{preset_name}")
def get_preset(preset_name: str, request: Request) -> dict[str, Any]:
    """Retrieve configuration for a specific preset."""
    root = _get_project_root(request)
    service = TrainingService(project_root=root)
    preset = service.get_preset(preset_name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    return preset
