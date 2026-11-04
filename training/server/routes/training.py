"""Training, evaluation, and export job dispatch endpoints."""

from __future__ import annotations

import logging
import unittest
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.kaggle_accounts import authenticate_kaggle_user, fetch_user_kernels
from training.registry.model_specs import get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.server.routes.training")

router = APIRouter(prefix="/training", tags=["training"])


class TrainRequest(BaseModel):
    """Payload for submitting training job."""

    model: str = Field(..., description="Target model key")
    epochs: int | None = Field(None, description="Total epochs override")
    batch_size: int | None = Field(None, description="Batch size override")
    lr: float | None = Field(None, description="Learning rate override")
    preset: str | None = Field(None, description="Preset configuration name")
    env: str = Field("local", description="Target environment ('local', 'kaggle', 'colab')")
    clean: bool = Field(False, description="Wipe local checkpoints and start fresh")
    auto_sync: bool = Field(False, description="Auto sync checkpoints to cloud")
    parallel: str = Field("auto", description="Parallel strategy ('auto', 'single', 'dp', 'ddp')")


class EvalRequest(BaseModel):
    """Payload for submitting evaluation job."""

    model: str = Field(..., description="Target model key")
    checkpoint_path: str | None = Field(None, description="Path to checkpoint (.pth)")
    batch_size: int | None = Field(None, description="Evaluation batch size")
    env: str = Field("local", description="Environment")


class ExportRequest(BaseModel):
    """Payload for submitting export job."""

    model: str = Field(..., description="Target model key")
    checkpoint_path: str | None = Field(None, description="Path to checkpoint")
    output_dir: str | None = Field(None, description="Export output directory")
    targets: list[str] | None = Field(None, description="Subset of export targets")


@router.post("/train")
def enqueue_training(payload: TrainRequest, request: Request) -> dict[str, Any]:
    """Enqueue background training job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="train", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Training job enqueued for {payload.model}",
    }


@router.post("/evaluate")
def enqueue_evaluation(payload: EvalRequest, request: Request) -> dict[str, Any]:
    """Enqueue background evaluation job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="eval", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Evaluation job enqueued for {payload.model}",
    }


@router.post("/export")
def enqueue_export(payload: ExportRequest, request: Request) -> dict[str, Any]:
    """Enqueue background export job."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="export", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Export job enqueued for {payload.model}",
    }


class UnitTestAllRequest(BaseModel):
    """Payload for submitting single-epoch diagnostic unit test across all registered models."""

    epochs: int = Field(1, description="Diagnostic epochs per model (default: 1)")
    env: str = Field("local", description="Execution environment ('local', 'kaggle')")
    force: bool = Field(False, description="Force re-train even if SOTA checkpoint exists")


@router.post("/unit-test-all")
def enqueue_unit_test_all(payload: UnitTestAllRequest, request: Request) -> dict[str, Any]:
    """Enqueue single-epoch diagnostic pass across all fleet models."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="test_all", model_key="all", params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Fleet diagnostic unit-test job enqueued for all models",
    }


# ─── Kaggle Cloud Engine Endpoints ───────────────────────────────────────────

_UNKNOWN_MODEL: dict[int | str, dict[str, Any]] = {
    404: {"description": "Model key not found in registry"}
}


def _require_registered_model(model_key: str) -> None:
    """Reject a model key the registry does not know (404) before any job or file is created."""
    if get_model_registry().get_model(model_key) is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")


class KaggleTrainRequest(BaseModel):
    """Payload for launching Kaggle cloud training."""

    model: str = Field(..., description="Target model key")
    gpu: str = Field("T4", description="Kaggle accelerator ('T4', 'P100', 'None')")
    auto_pull: bool = Field(True, description="Auto-pull checkpoints to LemGendaryModels per epoch")
    poll_interval: int = Field(5, description="Telemetry poll interval in seconds")


class KaggleMonitorRequest(BaseModel):
    """Payload for connecting to active Kaggle kernel stream."""

    kernel_slug: str = Field(..., description="Kaggle kernel slug (e.g. username/kernel-slug)")
    model: str | None = Field(None, description="Optional associated model key for auto-pull")
    auto_pull: bool = Field(False, description="Enable auto-pull on epoch completion")
    poll_interval: int = Field(5, description="Telemetry poll interval in seconds")


class KagglePullRequest(BaseModel):
    """Payload for downloading model variations from Kaggle Models Hub."""

    model: str = Field(..., description="Target model key")
    variation: str = Field(
        "all",
        description=(
            "Variation to pull ('all', 'pytorch', 'pytorch_fp32', 'onnx_fp16', 'onnx_fp32')"
        ),
    )
    force_download: bool = Field(
        False, description="Force re-download even if cached version is current"
    )
    username: str | None = Field(None, description="Optional Kaggle username override")


class KagglePushRequest(BaseModel):
    """Payload for uploading model checkpoints to Kaggle Models."""

    model: str = Field(..., description="Target model key")
    source_dir: str | None = Field(None, description="Optional custom source directory")
    username: str | None = Field(None, description="Optional Kaggle username override")


@router.get("/kaggle/status")
def get_kaggle_status() -> dict[str, Any]:
    """Check Kaggle credentials configuration and authentication status."""
    user, token = resolve_kaggle_credentials()
    authenticated = False
    if user and token:
        try:
            api = authenticate_kaggle_user(user, token)
            authenticated = api is not None
        except (ImportError, *KAGGLE_API_ERRORS) as exc:
            logger.warning("Kaggle authentication check failed for %s: %s", user, exc)
            authenticated = False

    return {
        "authenticated": authenticated,
        "username": user or "",
        "token_configured": bool(token),
    }


@router.get("/kaggle/kernels")
def list_kaggle_kernels(limit: int = 20) -> list[dict[str, Any]]:
    """List recent and active Kaggle kernels for authenticated user."""
    user, token = resolve_kaggle_credentials()
    if not user or not token:
        return []

    api = authenticate_kaggle_user(user, token)
    if not api:
        return []

    return fetch_user_kernels(api, user, limit=limit)


@router.post("/kaggle/train", responses=_UNKNOWN_MODEL)
def launch_kaggle_train(payload: KaggleTrainRequest, request: Request) -> dict[str, Any]:
    """Enqueue and launch Kaggle cloud training with live WebSocket telemetry."""
    _require_registered_model(payload.model)
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="kaggle_train", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Kaggle Cloud training queued for {payload.model} (GPU: {payload.gpu})",
        "model": payload.model,
    }


@router.post("/kaggle/monitor", responses=_UNKNOWN_MODEL)
def monitor_kaggle_kernel(payload: KaggleMonitorRequest, request: Request) -> dict[str, Any]:
    """Connect live WebSocket telemetry to an active Kaggle kernel."""
    if payload.model:
        _require_registered_model(payload.model)
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


@router.post("/kaggle/pull", responses=_UNKNOWN_MODEL)
def pull_kaggle_model_artifacts(payload: KagglePullRequest, request: Request) -> dict[str, Any]:
    """Enqueue a job that downloads all 4 model variations from Kaggle Models Hub."""
    _require_registered_model(payload.model)
    manager = request.app.state.job_manager
    params: dict[str, Any] = {
        "variation": payload.variation,
        "force_download": payload.force_download,
        "username": payload.username,
    }
    job_id = manager.submit_job(job_type="kaggle_pull", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "model": payload.model,
        "message": (
            f"Kaggle pull enqueued for {payload.model} "
            "(pytorch/default, pytorch/fp32, onnx/fp16, onnx/fp32)"
        ),
    }


@router.post("/kaggle/push", responses=_UNKNOWN_MODEL)
def push_kaggle_model_artifacts(payload: KagglePushRequest, request: Request) -> dict[str, Any]:
    """Enqueue a job staging and pushing all 4 variations of one model to Kaggle Models."""
    _require_registered_model(payload.model)
    manager = request.app.state.job_manager
    params: dict[str, Any] = {"variation": "all", "dry_run": False, "username": payload.username}
    job_id = manager.submit_job(job_type="kaggle_push", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "model": payload.model,
        "message": (
            f"Kaggle push enqueued for {payload.model} "
            "(pytorch/default, pytorch/fp32, onnx/fp16, onnx/fp32)"
        ),
    }


class KagglePushModelRequest(BaseModel):
    """Payload for pushing validated isolated variations to Kaggle Models Hub."""

    model: str = Field(..., description="Target model key")
    variation: str = Field(
        "all", description="Variation to push ('all', 'pytorch', 'onnx_fp32', 'onnx_fp16')"
    )
    version_notes: str | None = Field(None, description="Optional release version notes")
    dry_run: bool = Field(
        False, description="Validate payload and isolation without publishing to Kaggle"
    )


@router.post("/kaggle/push-model", responses=_UNKNOWN_MODEL)
def push_kaggle_model_variations(payload: KagglePushModelRequest) -> dict[str, Any]:
    """Publish validated isolated model variation(s) to Kaggle Models Hub."""
    _require_registered_model(payload.model)
    sync_mgr = KaggleModelSyncManager()
    result = sync_mgr.push_model(
        model_key=payload.model,
        variation=payload.variation,
        version_notes=payload.version_notes,
        dry_run=payload.dry_run,
    )
    return result.to_dict()


class KagglePushAllRequest(BaseModel):
    """Payload for staging and pushing all registered models to Kaggle Models Hub."""

    variation: str = Field(
        "all", description="Variation to push ('all', 'pytorch', 'onnx_fp32', 'onnx_fp16')"
    )
    version_notes: str | None = Field(None, description="Optional release version notes")
    dry_run: bool = Field(
        False, description="Validate payload and isolation without publishing to Kaggle"
    )


@router.post("/kaggle/push-all")
def push_all_kaggle_models(payload: KagglePushAllRequest, request: Request) -> dict[str, Any]:
    """Enqueue background fleet synchronization to push all models to Kaggle Models Hub."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="kaggle_push_all", model_key="all_models", params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": (
            "Kaggle fleet push enqueued for all models "
            f"(variation: {payload.variation}, dry_run: {payload.dry_run})"
        ),
        "variation": payload.variation,
        "dry_run": payload.dry_run,
    }


class LLMTrainRequest(BaseModel):
    """Payload for dispatching QLoRA instruction fine-tuning for assistant LLM."""

    model: str = Field("ai_helper_lemgend", description="Target LLM model key")
    epochs: int = Field(3, description="Training epochs")
    batch_size: int = Field(2, description="Per-device micro batch size")
    lr: float = Field(0.0002, description="Learning rate")
    env: str = Field("local", description="Training environment ('local', 'kaggle', 'colab')")


@router.post("/llm/train")
def enqueue_llm_training(payload: LLMTrainRequest, request: Request) -> dict[str, Any]:
    """Enqueue background QLoRA instruction tuning job for assistant LLM."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="train", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"LLM instruction tuning job enqueued for {payload.model}",
        "config": params,
    }


class LLMExportRequest(BaseModel):
    """Payload for compiling and exporting tri-format LLM binaries."""

    model: str = Field("ai_helper_lemgend", description="Target LLM model key")
    checkpoint: str | None = Field(None, description="Path to checkpoint file")
    output_dir: str | None = Field(None, description="Output directory for exported binaries")
    targets: list[str] | None = Field(
        None, description="Target export formats (e.g. ['adapter', 'merged', 'gguf'])"
    )


@router.post("/llm/export")
def enqueue_llm_export(payload: LLMExportRequest, request: Request) -> dict[str, Any]:
    """Enqueue background tri-format export job for an LLM."""
    manager = request.app.state.job_manager
    params = payload.model_dump()
    job_id = manager.submit_job(job_type="export", model_key=payload.model, params=params)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"LLM export job enqueued for {payload.model}",
        "config": params,
    }


class UnitTestsRunRequest(BaseModel):
    """Payload for executing unit test suites."""

    pattern: str = Field("test_*.py", description="Test discovery file pattern")


@router.post("/unit-test-run")
def run_unit_tests(payload: UnitTestsRunRequest) -> dict[str, Any]:
    """Execute unit test suite in-process and return structured test outcome."""
    project_root = get_project_root()
    tests_dir = project_root / "tests" / "unit"
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=payload.pattern)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures_count": len(result.failures),
        "errors_count": len(result.errors),
        "failures": [str(f[0]) for f in result.failures],
        "errors": [str(e[0]) for e in result.errors],
    }
