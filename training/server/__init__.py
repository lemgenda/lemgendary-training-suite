"""LemGendary Model Training Suite Sidecar Server Subsystem."""

from typing import Any

from training.server.jobs import JobManager
from training.server.state import ServerState

__all__ = ["JobManager", "ServerState", "app", "create_app"]


def __getattr__(name: str) -> Any:
    """Lazily import app and create_app on access."""
    if name in ("app", "create_app"):
        from training.server.app import create_app as _create_app, get_app as _get_app
        if name == "create_app":
            return _create_app
        return _get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

