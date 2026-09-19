"""Persistent job queue management endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status ('pending', 'running', 'completed', 'failed', 'cancelled')"),
    model_key: Optional[str] = Query(None, description="Filter by target model key"),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """List persistent jobs from SQLite database."""
    state = request.app.state.server_state
    return state.list_jobs(status=status, model_key=model_key, limit=limit)


@router.get("/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Retrieve detailed state and metrics of a job."""
    state = request.app.state.server_state
    job = state.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    """Cancel an active or pending background job."""
    manager = request.app.state.job_manager
    cancelled = manager.cancel_job(job_id)
    if not cancelled:
        job = request.app.state.server_state.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return {"job_id": job_id, "status": job["status"], "cancelled": False, "message": "Job cannot be cancelled in its current state."}
    return {"job_id": job_id, "status": "cancelled", "cancelled": True}


@router.get("/{job_id}/logs")
def get_job_logs(job_id: str, request: Request, tail: int = Query(200, ge=1, le=5000)) -> dict[str, Any]:
    """Read buffered log output from disk."""
    state = request.app.state.server_state
    job = state.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    log_file = Path(job["log_file"]) if job.get("log_file") else state.logs_dir / f"{job_id}.log"
    if not log_file.exists():
        return {"job_id": job_id, "lines": [], "total_lines": 0}

    try:
        lines = log_file.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-tail:] if len(lines) > tail else lines
        return {"job_id": job_id, "lines": tail_lines, "total_lines": len(lines)}
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed reading log file: {e}")
