"""LemGendary Model Training Suite Sidecar Server Subsystem.

Exports are resolved lazily so that light submodules (for example
``training.server.security``) import without the training stack.
"""

from typing import Any

__all__ = ["JobManager", "ServerState", "app", "create_app"]


def __getattr__(name: str) -> Any:
    """Lazily import the public server objects on access."""
    if name == "JobManager":
        from training.server.jobs import JobManager as _JobManager

        return _JobManager
    if name == "ServerState":
        from training.server.state import ServerState as _ServerState

        return _ServerState
    if name in ("app", "create_app"):
        from training.server.app import create_app as _create_app
        from training.server.app import get_app as _get_app

        if name == "create_app":
            return _create_app
        return _get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
