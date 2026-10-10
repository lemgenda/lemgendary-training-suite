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


# ─── Kaggle Cloud Engine Endpoints ───────────────────────────────────────────

class KaggleTrainRequest(BaseModel):
    """Payload for launching Kaggle cloud training."""
    model: str = Field(..., description="Target model key")
    gpu: str = Field("T4", description="Kaggle accelerator ('T4', 'P100', 'None')")
    auto_pull: bool = Field(True, description="Auto-pull checkpoints to LemGendaryModels per epoch")
    poll_interval: int = Field(5, description="Telemetry poll interval in seconds")


class KaggleMonitorRequest(BaseModel):
    """Payload for connecting to active Kaggle kernel stream."""
    kernel_slug: str = Field(..., description="Kaggle kernel slug (e.g. username/kernel-slug)")
    model: Optional[str] = Field(None, description="Optional associated model key for auto-pull")
    auto_pull: bool = Field(False, description="Enable auto-pull on epoch completion")
    poll_interval: int = Field(5, description="Telemetry poll interval in seconds")


class KagglePullRequest(BaseModel):
    """Payload for downloading model checkpoints from Kaggle Models."""
    model: str = Field(..., description="Target model key")


class KagglePushRequest(BaseModel):
    """Payload for uploading model checkpoints to Kaggle Models."""
    model: str = Field(..., description="Target model key")
    source_dir: Optional[str] = Field(None, description="Optional custom source directory")
    username: Optional[str] = Field(None, description="Optional Kaggle username override")


@router.get("/kaggle/status")
def get_kaggle_status() -> dict[str, Any]:
    """Check Kaggle credentials configuration and authentication status."""
    from training.cloud.credentials import resolve_kaggle_credentials
    from training.kaggle_monitor import authenticate_kaggle_user

    user, token = resolve_kaggle_credentials()
    authenticated = False
    if user and token:
        try:
            api = authenticate_kaggle_user(user, token)
            authenticated = bool(api is not None)
        except Exception:
            authenticated = False

    return {
        "authenticated": authenticated,
        "username": user or "",
        "token_configured": bool(token),
    }


@router.get("/kaggle/kernels")
def list_kaggle_kernels(request: Request, limit: int = 20) -> list[dict[str, Any]]:
    """List recent and active Kaggle kernels for authenticated user."""
    from training.cloud.credentials import resolve_kaggle_credentials
    from training.kaggle_monitor import authenticate_kaggle_user, fetch_user_kernels

    user, token = resolve_kaggle_credentials()
    if not user or not token:
        return []

    api = authenticate_kaggle_user(user, token)
    if not api:
        return []

    return fetch_user_kernels(api, user, limit=limit)


@router.post("/kaggle/train")
def launch_kaggle_train(payload: KaggleTrainRequest, request: Request) -> dict[str, Any]:
    """Enqueue and launch Kaggle cloud training with live WebSocket telemetry."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="kaggle_train", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Kaggle Cloud training queued for {payload.model} (GPU: {payload.gpu})",
        "model": payload.model,
    }


@router.post("/kaggle/monitor")
def monitor_kaggle_kernel(payload: KaggleMonitorRequest, request: Request) -> dict[str, Any]:
    """Connect live WebSocket telemetry to an active Kaggle kernel."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    target_key = payload.model or payload.kernel_slug.split("/")[-1]
    job_id = manager.submit_job(job_type="kaggle_monitor", model_key=target_key, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Connected telemetry to Kaggle kernel {payload.kernel_slug}",
        "kernel_slug": payload.kernel_slug,
    }


@router.post("/kaggle/pull")
def pull_kaggle_model_artifacts(payload: KagglePullRequest) -> dict[str, Any]:
    """Pull latest model weights and metrics from Kaggle Models to LemGendaryModels."""
    from training.kaggle_cloud_manager import pull_kaggle_artifacts

    success = pull_kaggle_artifacts(payload.model)
    return {
        "status": "success" if success else "failed",
        "model": payload.model,
        "message": f"Artifact pull {'completed' if success else 'failed'} for {payload.model}",
    }


@router.post("/kaggle/push")
def push_kaggle_model_artifacts(payload: KagglePushRequest) -> dict[str, Any]:
    """Stage and upload local model checkpoints and metrics to Kaggle Models."""
    from training.kaggle_cloud_manager import push_kaggle_artifacts

    success = push_kaggle_artifacts(
        model_name=payload.model,
        source_dir=payload.source_dir,
        username=payload.username,
    )
    return {
        "status": "success" if success else "failed",
        "model": payload.model,
        "message": f"Artifact push {'completed successfully' if success else 'failed'} for {payload.model}",
    }
