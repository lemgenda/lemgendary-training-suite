"""Checkpoint management, recovery, and metric history tracking package."""

from training.checkpoint.lifecycle_manager import CheckpointLifecycleManager

def generate_preflight_assets(*args, **kwargs):
    from training.checkpoint.preflight_generator import generate_preflight_assets as _fn

    return _fn(*args, **kwargs)


def generate_all_preflight_assets(*args, **kwargs):
    from training.checkpoint.preflight_generator import generate_all_preflight_assets as _fn

    return _fn(*args, **kwargs)

from training.checkpoint.manager import (
    CheckpointSaveError,
    audit_disk_space,
    safe_atomic_save,
    safe_load_checkpoint,
)
from training.checkpoint.recovery import CheckpointRecoveryEngine
from training.checkpoint.resume import (
    ResumeState,
    parse_resume_state,
    scale_resume_progress,
    stretch_scheduler_runway,
)
from training.checkpoint.vault import MetricRecord, MetricVault

__all__ = [
    "CheckpointLifecycleManager",
    "CheckpointRecoveryEngine",
    "CheckpointSaveError",
    "MetricRecord",
    "MetricVault",
    "ResumeState",
    "audit_disk_space",
    "generate_all_preflight_assets",
    "generate_preflight_assets",
    "parse_resume_state",
    "safe_atomic_save",
    "safe_load_checkpoint",
    "scale_resume_progress",
    "stretch_scheduler_runway",
]

