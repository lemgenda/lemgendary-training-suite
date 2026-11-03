"""Real-time WebSocket streaming routes for jobs and global log telemetry."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("lemtrain.server.routes.ws")

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/jobs/{job_id}/logs")
async def stream_job_logs(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time log events for a specific running job."""
    await websocket.accept()
    manager = websocket.app.state.job_manager
    queue = manager.subscribe_job_logs(job_id)

    try:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect as exc:
        logger.debug("Job log WebSocket for %s disconnected (code=%s)", job_id, exc.code)
    except asyncio.CancelledError:
        logger.debug("Job log WebSocket stream for %s cancelled", job_id)
        raise
    finally:
        manager.unsubscribe_job_logs(job_id, queue)


@router.websocket("/logs")
async def stream_global_logs(websocket: WebSocket) -> None:
    """Stream all daemon event logs."""
    await websocket.accept()
    manager = websocket.app.state.job_manager
    queue = manager.subscribe_global_logs()

    try:
        while True:
            msg = await queue.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect as exc:
        logger.debug("Global log WebSocket disconnected (code=%s)", exc.code)
    except asyncio.CancelledError:
        logger.debug("Global log WebSocket stream cancelled")
        raise
    finally:
        manager.unsubscribe_global_logs(queue)
