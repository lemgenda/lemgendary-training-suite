"""FastAPI Sidecar Daemon Application Factory and Lifespan Manager.

Runs on port 8200, exposing REST and WebSocket endpoints for background training,
evaluations, checkpoint telemetry, and GUI integration.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from collections.abc import AsyncGenerator
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from training.server.jobs import JobManager
from training.server.routes import (
    config as config_route,
    datasets as datasets_route,
    env as env_route,
    gui as gui_route,
    health as health_route,
    jobs as jobs_route,
    models as models_route,
    training as training_route,
    ws as ws_route,
)
from training.server.state import ServerState
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.server")

security_bearer = HTTPBearer(auto_error=False)
security_header = APIKeyHeader(name="X-LemTrain-Token", auto_error=False)


async def verify_auth_token(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Security(security_bearer),
    api_key: str | None = Security(security_header),
) -> bool:
    """Validate bearer or header token against local server secret."""
    # Allow open paths
    path = request.url.path
    if path in ("/api/health", "/docs", "/openapi.json", "/redoc") or path.startswith("/api/ws/"):
        return True

    state: ServerState = request.app.state.server_state
    token = bearer.credentials if bearer else api_key

    if not token or not state.verify_token(token):
        # Check if running in relaxed local-dev mode without tokens
        if not getattr(request.app.state, "enforce_auth", True):
            return True
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid LemTrain authentication token",
        )
    return True


def create_app(project_root: Path | None = None, enforce_auth: bool = False) -> FastAPI:
    """Create and configure FastAPI application instance."""
    root = project_root or get_project_root()
    server_state = ServerState(project_root=root)
    job_manager = JobManager(state=server_state)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        # Startup sequence
        logger.info("Initializing LemGendary Training Suite Sidecar Daemon on port 8200...")
        server_state.write_pid()
        token = server_state.get_or_create_token()
        recovered = server_state.recover_orphaned_jobs()
        if recovered > 0:
            logger.warning("Recovered %d orphaned jobs from previous daemon run.", recovered)

        job_manager.set_event_loop(asyncio.get_running_loop())
        logger.info("Daemon ready. Local token: %s...", token[:8])

        yield

        # Shutdown sequence
        logger.info("Shutting down LemGendary Training Suite Sidecar Daemon...")
        job_manager.shutdown()
        server_state.clear_pid()
        logger.info("Daemon terminated cleanly.")

    openapi_tags = [
        {"name": "health", "description": "Daemon health, uptime, and accelerator status."},
        {"name": "config", "description": "Configuration and canonical training presets."},
        {"name": "jobs", "description": "Persistent asynchronous job queue and log streaming."},
        {"name": "models", "description": "Model registry, parameters, and architecture audit."},
        {"name": "training", "description": "In-process training, evaluation, and export job dispatch."},
        {"name": "datasets", "description": "Compiled datasets and manifold exploration."},
        {"name": "env", "description": "Hardware telemetry and host resources."},
        {"name": "gui", "description": "Desktop GUI dashboard aggregation and quick-dispatch endpoints."},
        {"name": "ws", "description": "Real-time WebSocket telemetry and event feeds."},
    ]

    application = FastAPI(
        title="LemGendary Model Training Suite Sidecar API",
        version="16.2.9",
        description="Local background daemon coordinating training runs, evaluations, exports, and telemetry.",
        openapi_tags=openapi_tags,
        lifespan=lifespan,
    )

    application.state.server_state = server_state
    application.state.job_manager = job_manager
    application.state.enforce_auth = enforce_auth

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach route modules
    dependencies = [Depends(verify_auth_token)] if enforce_auth else []

    application.include_router(health_route.router, prefix="/api")
    application.include_router(config_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(jobs_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(models_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(training_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(datasets_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(env_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(gui_route.router, prefix="/api", dependencies=dependencies)
    application.include_router(ws_route.router, prefix="/api")

    return application


app = create_app()
