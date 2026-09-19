"""Training, evaluation, and export job dispatch endpoints."""

from __future__ import annotations

from typing import Any, List, Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/training", tags=["training"])


class TrainRequest(BaseModel):
    """Payload for submitting training job."""
    model: str = Field(..., description="Target model key")
    epochs: Optional[int] = Field(None, description="Total epochs override")
    batch_size: Optional[int] = Field(None, description="Batch size override")
    lr: Optional[float] = Field(None, description="Learning rate override")
    preset: Optional[str] = Field(None, description="Preset configuration name")
    env: str = Field("local", description="Target environment ('local', 'kaggle', 'colab')")
    clean: bool = Field(False, description="Wipe local checkpoints and start fresh")
    auto_sync: bool = Field(False, description="Auto sync checkpoints to cloud")
    parallel: str = Field("auto", description="Parallel strategy ('auto', 'single', 'dp', 'ddp')")


class EvalRequest(BaseModel):
    """Payload for submitting evaluation job."""
    model: str = Field(..., description="Target model key")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint (.pth)")
    batch_size: Optional[int] = Field(None, description="Evaluation batch size")
    env: str = Field("local", description="Environment")


class ExportRequest(BaseModel):
    """Payload for submitting export job."""
    model: str = Field(..., description="Target model key")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint")
    output_dir: Optional[str] = Field(None, description="Export output directory")
    targets: Optional[List[str]] = Field(None, description="Subset of export targets")


@router.post("/train")
def enqueue_training(payload: TrainRequest, request: Request) -> dict[str, Any]:
    """Enqueue background training job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="train", model_key=payload.model, params=params)
    return {"job_id": job_id, "status": "pending", "message": f"Training job enqueued for {payload.model}"}


@router.post("/evaluate")
def enqueue_evaluation(payload: EvalRequest, request: Request) -> dict[str, Any]:
    """Enqueue background evaluation job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="eval", model_key=payload.model, params=params)
    return {"job_id": job_id, "status": "pending", "message": f"Evaluation job enqueued for {payload.model}"}


@router.post("/export")
def enqueue_export(payload: ExportRequest, request: Request) -> dict[str, Any]:
    """Enqueue background export job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="export", model_key=payload.model, params=params)
    return {"job_id": job_id, "status": "pending", "message": f"Export job enqueued for {payload.model}"}
