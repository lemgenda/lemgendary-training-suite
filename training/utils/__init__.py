"""Core utility primitives for LemGendary Model Training Suite."""

from training.utils.logging import ForceTTY, setup_logging
from training.utils.paths import get_project_root, get_workspace_root, resolve_path, bootstrap_sys_path
from training.utils.subprocess import run_command, SubprocessExecutionError
from training.utils.interrupt import (
    register_emergency_sync,
    register_active_process,
    cleanup_active_processes,
    install_signal_handlers,
    silent_worker_excepthook,
)
from training.utils.env_delegate import EnvManagerDelegate
from training.utils.dataset_delegate import DatasetCompilerDelegate

__all__ = [
    "ForceTTY",
    "setup_logging",
    "get_project_root",
    "get_workspace_root",
    "resolve_path",
    "bootstrap_sys_path",
    "run_command",
    "SubprocessExecutionError",
    "register_emergency_sync",
    "register_active_process",
    "cleanup_active_processes",
    "install_signal_handlers",
    "silent_worker_excepthook",
    "EnvManagerDelegate",
    "DatasetCompilerDelegate",
]
