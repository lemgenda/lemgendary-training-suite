"""LemGendary Model Training Suite Sidecar Server Subsystem.

Exports are resolved lazily so that light submodules (for example
``training.server.security``) import without the training stack.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.server.app import create_app
    from training.server.jobs import JobManager
    from training.server.state import ServerState

__all__ = ["JobManager", "ServerState", "app", "create_app"]


# Public name -> submodule that defines it (PEP 562 lazy exports).
_LAZY_EXPORTS = {
    "JobManager": "jobs",
    "ServerState": "state",
    "create_app": "app",
}


def __getattr__(name: str) -> Any:
    """Lazily import the public server objects on access."""
    if name == "app":
        return importlib.import_module(f"{__name__}.app").get_app()
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
