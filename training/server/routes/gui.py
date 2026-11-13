"""Desktop GUI aggregation endpoints."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.server.routes.gui_model_stats import model_with_stats
from training.server.routes.gui_schemas import ModelWithStats
from training.server.routes.gui_status import UNKNOWN_VERDICT
from training.server.routes.models import _load_registry
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.training_service import TrainingService
from training.utils.paths import get_models_hub_root, get_project_root

logger = logging.getLogger("lemtrain.server.routes.gui")

router = APIRouter(prefix="/gui", tags=["gui"])


class QuickTrainRequest(BaseModel):
    """Payload for quick train trigger."""

    model_key: str = Field(..., description="Target model key")
    preset: str = Field("quick-sota", description="Preset name from presets.yaml")
    clean: bool = Field(False, description="Wipe checkpoints and start fresh")
    override_frozen: bool = Field(
        False, description="Resume although a frozen run setting changed (logged and recorded)"
    )
    epochs: int | None = Field(None, description="Optional override for training epochs")
    batch_size: int | None = Field(None, description="Optional override for batch size")
    learning_rate: float | None = Field(None, description="Optional override for learning rate")
    env: str = Field("local", description="Execution environment ('local', 'kaggle', 'colab')")
    ladder_stage: int | None = Field(
        None, description="Selected spatial ladder resolution or timeframe minute stage"
    )
    enable_sawtooth: bool | None = Field(
        True, description="Enable Sawtooth VRAM Governor memory guard"
    )


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("/state")
async def get_gui_state(request: Request) -> dict[str, Any]:
    """Retrieve unified state snapshot for Desktop GUI dashboard."""
    root = _get_root(request)
    state = request.app.state.server_state
    audit_svc = AuditService(project_root=root)
    train_svc = TrainingService(project_root=root)

    loop = asyncio.get_running_loop()
    system_info = await loop.run_in_executor(None, audit_svc.audit_system)
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


_STATS_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
)
"""Failures of one malformed registry entry; the entry is still listed, as 'unknown'."""


def _unknown_entry(model_key: str, info: Any, exc: Exception) -> ModelWithStats:
    """Entry for a model whose stats could not be computed: no verdict is made up."""
    name = info.get("name") if isinstance(info, dict) else None
    name = name if isinstance(name, str) else model_key
    return ModelWithStats(
        model_key=model_key,
        key=model_key,
        name=name,
        display_name=name,
        training_status=UNKNOWN_VERDICT.training_status,
        is_fully_trained=UNKNOWN_VERDICT.is_fully_trained,
        stats_error=f"{type(exc).__name__}: {exc}",
    )


@router.get("/models/with-stats", response_model=list[ModelWithStats])
def get_models_with_stats(request: Request) -> list[ModelWithStats]:
    """Retrieve model architectures paired with checkpoint existence and size.

    Every entry carries ``training_status`` and ``is_fully_trained``. A
    registry entry whose stats cannot be computed is still listed, with
    status ``unknown`` and the reason in ``stats_error``.
    """
    root = _get_root(request)
    registry = _load_registry(root)
    ckpt_svc = CheckpointService(project_root=root)
    checkpoints = ckpt_svc.list_checkpoints()

    ckpts_by_model: dict[str, list[dict[str, Any]]] = {}
    for c in checkpoints:
        ckpts_by_model.setdefault(c["model_key"], []).append(c)

    hub_root = get_models_hub_root(root)

    results: list[ModelWithStats] = []
    for model_key, info in registry.items():
        try:
            entry = model_with_stats(
                model_key, info, root, hub_root, ckpts_by_model.get(model_key, [])
            )
        except _STATS_ERRORS as exc:
            logger.warning("Could not compute GUI stats for %s: %s", model_key, exc)
            entry = _unknown_entry(model_key, info, exc)
        results.append(entry)

    return sorted(results, key=lambda m: m.model_key)


@router.post("/quick-train")
def quick_train(payload: QuickTrainRequest, request: Request) -> dict[str, Any]:
    """Dispatch fast one-click training job using a named preset."""
    root = _get_root(request)
    train_svc = TrainingService(project_root=root)
    preset_cfg = train_svc.get_preset(payload.preset)
    if not preset_cfg:
        raise HTTPException(status_code=400, detail=f"Invalid preset '{payload.preset}'")

    registry = _load_registry(root)
    model_cfg = registry.get(payload.model_key, {})

    resolved_epochs = payload.epochs
    if resolved_epochs is None and model_cfg.get("epochs"):
        try:
            resolved_epochs = int(model_cfg["epochs"])
        except (ValueError, TypeError):
            resolved_epochs = None
    if resolved_epochs is None:
        resolved_epochs = preset_cfg.get("epochs")

    resolved_batch = payload.batch_size
    if resolved_batch is None and model_cfg.get("batch_size") is not None:
        cfg_bs = model_cfg.get("batch_size")
        if isinstance(cfg_bs, int) and cfg_bs > 0:
            resolved_batch = cfg_bs
        elif isinstance(cfg_bs, str) and cfg_bs.strip().isdigit() and int(cfg_bs) > 0:
            resolved_batch = int(cfg_bs)
    if resolved_batch is None:
        resolved_batch = preset_cfg.get("batch_size")

    resolved_lr = payload.learning_rate
    if resolved_lr is None and model_cfg.get("learning_rate") is not None:
        try:
            resolved_lr = float(model_cfg["learning_rate"])
        except (ValueError, TypeError):
            resolved_lr = None
    if resolved_lr is None:
        resolved_lr = preset_cfg.get("learning_rate")

    manager = request.app.state.job_manager
    params: dict[str, Any] = {
        "model": payload.model_key,
        "preset": payload.preset,
        "epochs": resolved_epochs,
        "batch_size": resolved_batch,
        "learning_rate": resolved_lr,
        "clean": payload.clean,
        "override_frozen": payload.override_frozen,
        "env": payload.env,
        "resolution": payload.ladder_stage,
        "enable_sawtooth": payload.enable_sawtooth if payload.enable_sawtooth is not None else True,
    }
    if payload.model_key == "forex_predictor":
        params["task_type"] = "forex"
    job_id = manager.submit_job(job_type="train", model_key=payload.model_key, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "preset": payload.preset,
        "params": params,
    }
