"""LemGendary Training Suite Core Training Engine Subpackage.

Provides modular execution contexts, AMP management, optimizer/scheduler builders,
epoch runners, validation passes, and the central engine coordinator.
"""

from __future__ import annotations

from training.training.amp import (
    GradClip,
    compute_ssim_gpu,
    create_grad_scaler,
    get_autocast_context,
    safe_backward,
)
from training.training.context import (
    DataComponents,
    GovernanceComponents,
    HardwareComponents,
    ModelsHubPaths,
    ModelStack,
    ResumeGuards,
    RunSettings,
    RunState,
    TrainingContext,
    TrainingPaths,
)
from training.training.engine import TrainingSummary, run_training
from training.training.epoch import train_one_epoch
from training.training.optimizer import build_optimizer, build_scheduler
from training.training.validation import validate_one_epoch

__all__ = [
    "DataComponents",
    "GovernanceComponents",
    "GradClip",
    "HardwareComponents",
    "ModelStack",
    "ModelsHubPaths",
    "ResumeGuards",
    "RunSettings",
    "RunState",
    "TrainingContext",
    "TrainingPaths",
    "TrainingSummary",
    "build_optimizer",
    "build_scheduler",
    "compute_ssim_gpu",
    "create_grad_scaler",
    "get_autocast_context",
    "run_training",
    "safe_backward",
    "train_one_epoch",
    "validate_one_epoch",
]
