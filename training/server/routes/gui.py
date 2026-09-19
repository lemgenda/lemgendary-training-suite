"""Desktop GUI aggregation endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.server.routes.models import _load_registry
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.training_service import TrainingService
from training.utils.paths import get_project_root

router = APIRouter(prefix="/gui", tags=["gui"])


class QuickTrainRequest(BaseModel):
    """Payload for quick train trigger."""
    model_key: str = Field(..., description="Target model key")
    preset: str = Field("quick-sota", description="Preset name from presets.yaml")
    clean: bool = Field(False, description="Wipe checkpoints and start fresh")
    epochs: int | None = Field(None, description="Optional override for training epochs")
    batch_size: int | None = Field(None, description="Optional override for batch size")
    learning_rate: float | None = Field(None, description="Optional override for learning rate")
    env: str = Field("local", description="Execution environment ('local', 'kaggle', 'colab')")


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("/state")
def get_gui_state(request: Request) -> dict[str, Any]:
    """Retrieve unified state snapshot for Desktop GUI dashboard."""
    root = _get_root(request)
    state = request.app.state.server_state
    audit_svc = AuditService(project_root=root)
    train_svc = TrainingService(project_root=root)

    system_info = audit_svc.audit_system()
    presets = train_svc.load_presets()
    active_jobs = state.list_jobs(status="running", limit=10)
    pending_jobs = state.list_jobs(status="pending", limit=10)
    recent_jobs = state.list_jobs(limit=10)
    models = _load_registry(root)

    return {
        "status": "online",
        "system": system_info,
        "presets": presets,
        "models_count": len(models),
        "active_jobs_count": len(active_jobs),
        "pending_jobs_count": len(pending_jobs),
        "recent_jobs": recent_jobs,
    }


@router.get("/models/with-stats")
def get_models_with_stats(request: Request) -> list[dict[str, Any]]:
    """Retrieve model architectures paired with checkpoint existence and size."""
    root = _get_root(request)
    registry = _load_registry(root)
    ckpt_svc = CheckpointService(project_root=root)
    checkpoints = ckpt_svc.list_checkpoints()

    ckpts_by_model: dict[str, list[dict[str, Any]]] = {}
    for c in checkpoints:
        k = c["model_key"]
        if k not in ckpts_by_model:
            ckpts_by_model[k] = []
        ckpts_by_model[k].append(c)

    results: list[dict[str, Any]] = []
    for model_key, info in registry.items():
        model_ckpts = ckpts_by_model.get(model_key, [])
        best_ckpt = next((c for c in model_ckpts if c["is_best"]), None)
        latest_ckpt = max(model_ckpts, key=lambda c: c.get("epoch") or 0) if model_ckpts else None

        results.append({
            "model_key": model_key,
            "name": info.get("name", model_key),
            "category": info.get("category", "general"),
            "class_name": info.get("class_name"),
            "checkpoints_count": len(model_ckpts),
            "has_best_checkpoint": best_ckpt is not None,
            "best_checkpoint_size_mb": best_ckpt["size_mb"] if best_ckpt else None,
            "best_checkpoint_epoch": best_ckpt["epoch"] if best_ckpt else None,
            "latest_checkpoint_epoch": latest_ckpt["epoch"] if latest_ckpt else None,
            "resolution": info.get("resolution"),
        })

    return sorted(results, key=lambda x: x["model_key"])


@router.post("/quick-train")
def quick_train(payload: QuickTrainRequest, request: Request) -> dict[str, Any]:
    """Dispatch fast one-click training job using a named preset."""
    root = _get_root(request)
    train_svc = TrainingService(project_root=root)
    preset_cfg = train_svc.get_preset(payload.preset)
    if not preset_cfg:
        raise HTTPException(status_code=400, detail=f"Invalid preset '{payload.preset}'")

    manager = request.app.state.job_manager
    params: dict[str, Any] = {
        "model": payload.model_key,
        "preset": payload.preset,
        "epochs": payload.epochs if payload.epochs is not None else preset_cfg.get("epochs"),
        "batch_size": payload.batch_size if payload.batch_size is not None else preset_cfg.get("batch_size"),
        "learning_rate": payload.learning_rate if payload.learning_rate is not None else preset_cfg.get("learning_rate"),
        "clean": payload.clean,
        "env": payload.env,
    }
    job_id = manager.submit_job(job_type="train", model_key=payload.model_key, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "preset": payload.preset,
        "params": params,
    }

