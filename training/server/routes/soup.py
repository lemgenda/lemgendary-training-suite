"""Pareto Model Soup endpoint: ``POST /api/models/{model_key}/soup``.

Kept apart from ``routes/models.py`` so it does not import torch at module
load; the blend itself runs through ``training.model_soup.blend_model_soup``,
the same service the ``lemtrain soup`` CLI command uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from training.model_soup import SoupError, blend_model_soup
from training.utils.paths import get_project_root

router = APIRouter(prefix="/models", tags=["models"])


class SoupRequest(BaseModel):
    """Optional soup parameters. An empty body blends every vault checkpoint uniformly."""

    donors: list[str] | None = None
    weights: list[float] | None = None


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.post("/{model_key}/soup")
def blend_model_soup_route(model_key: str, request: Request, body: SoupRequest | None = None) -> dict[str, Any]:
    """Blend specialized vault checkpoints using Pareto Model Soup."""
    params = body or SoupRequest()
    try:
        result = blend_model_soup(
            model_key,
            project_root=_get_root(request),
            donors=params.donors,
            weights=params.weights,
        )
    except SoupError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    return {
        "model_key": model_key,
        "success": True,
        "soup_checkpoint": str(result.output_path),
        "donors": [p.name for p in result.donors],
        "weights": result.weights,
    }
