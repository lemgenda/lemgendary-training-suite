"""Notebook generation REST endpoints for GUI and compiler suite integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from training.services.notebook_service import NotebookService
from training.utils.paths import get_project_root

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


class NotebookGenerateRequest(BaseModel):
    """Payload for generating notebooks for a single model."""

    model_key: str = Field(..., description="Target model key (e.g. nima_aesthetic_mobile)")
    platform: str = Field("all", description="Execution platform: 'kaggle', 'colab', or 'all'")
    kinds: Optional[List[str]] = Field(
        None, description="Notebook kinds: ['training', 'inference', 'usage'] (defaults to all)"
    )
    output_dir: Optional[str] = Field(None, description="Custom target export directory")


class NotebookRefreshAllRequest(BaseModel):
    """Payload for generating notebooks for all registered models."""

    platform: str = Field("all", description="Execution platform: 'kaggle', 'colab', or 'all'")
    kinds: Optional[List[str]] = Field(
        None, description="Notebook kinds: ['training', 'inference', 'usage'] (defaults to all)"
    )
    output_dir: Optional[str] = Field(None, description="Custom target export directory")


@router.get("/models")
def list_notebook_models(request: Request) -> list[str]:
    """List all models registered and available for notebook generation."""
    root = (
        request.app.state.server_state.project_root
        if hasattr(request.app.state, "server_state")
        else get_project_root()
    )
    service = NotebookService(project_root=root)
    return service.list_supported_models()


@router.post("/generate")
def generate_notebooks_endpoint(
    payload: NotebookGenerateRequest,
    request: Request,
) -> dict[str, Any]:
    """Generate Jupyter training, inference, and usage notebooks for a specified model."""
    root = (
        request.app.state.server_state.project_root
        if hasattr(request.app.state, "server_state")
        else get_project_root()
    )
    service = NotebookService(project_root=root)
    try:
        results = service.generate_notebooks(
            model_key=payload.model_key,
            platform=payload.platform,
            kinds=payload.kinds,
            output_dir=Path(payload.output_dir) if payload.output_dir else None,
        )
        return {
            "success": True,
            "model_key": payload.model_key,
            "platform": payload.platform,
            "generated": results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate notebooks: {e}")


@router.post("/refresh-all")
def refresh_all_notebooks_endpoint(
    payload: NotebookRefreshAllRequest,
    request: Request,
) -> dict[str, Any]:
    """Batch generate Jupyter notebooks for all models in the unified models registry."""
    root = (
        request.app.state.server_state.project_root
        if hasattr(request.app.state, "server_state")
        else get_project_root()
    )
    service = NotebookService(project_root=root)
    try:
        all_results = service.generate_all_models(
            platform=payload.platform,
            kinds=payload.kinds,
            output_dir=Path(payload.output_dir) if payload.output_dir else None,
        )
        total_files = sum(len(v) for v in all_results.values())
        return {
            "success": True,
            "platform": payload.platform,
            "total_models": len(all_results),
            "total_files": total_files,
            "results": all_results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to batch generate notebooks: {e}")
