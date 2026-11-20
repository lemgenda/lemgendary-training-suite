"""Checkpoint management, recovery, and metric history tracking package."""

from training.checkpoint.lifecycle_manager import (
    CheckpointLifecycleManager,
    ExportSource,
    canonicalize_checkpoints,
)
from training.checkpoint.manager import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointMigrationError,
    CheckpointSaveError,
    audit_disk_space,
    safe_atomic_save,
    safe_load_checkpoint,
)
from training.checkpoint.preflight_generator import (
    generate_all_preflight_assets,
    generate_preflight_assets,
)
from training.checkpoint.recovery import CheckpointRecoveryEngine
from training.checkpoint.resume import (
    ResumeProgress,
    ResumeState,
    parse_resume_state,
    scale_resume_progress,
    stretch_scheduler_runway,
)
from training.checkpoint.vault import (
    MetricRecord,
    MetricVault,
    VaultConsistencyReport,
    prune_vault_checkpoints,
    validate_vault_consistency,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointLifecycleManager",
    "CheckpointMigrationError",
    "CheckpointRecoveryEngine",
    "CheckpointSaveError",
    "ExportSource",
    "MetricRecord",
    "MetricVault",
    "ResumeProgress",
    "ResumeState",
    "VaultConsistencyReport",
    "audit_disk_space",
    "canonicalize_checkpoints",
    "generate_all_preflight_assets",
    "generate_preflight_assets",
    "parse_resume_state",
    "prune_vault_checkpoints",
    "safe_atomic_save",
    "safe_load_checkpoint",
    "scale_resume_progress",
    "stretch_scheduler_runway",
    "validate_vault_consistency",
]
