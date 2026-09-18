"""Immutable Execution Context for LemGendary Training Suite.

Encapsulates all runtime components: hardware policy, models, optimizers,
schedulers, data loaders, governance mechanisms, and checkpoint managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from training.checkpoint import MetricVault, ResumeState
from training.governance import SmartTrainingGovernor, SotaTracker
from training.hardware.discovery import DeviceInfo
from training.hardware.policy import ExecutionPolicy
from training.hardware.sentinel import SentinelGuard


@dataclass(frozen=True)
class TrainingPaths:
    """Filesystem pathways for checkpoints, telemetry, exports, and Hub synchronizations."""

    project_root: Path
    local_checkpoint_dir: Path
    hub_checkpoint_dir: Path
    export_dir: Path
    progress_local_path: Path
    best_checkpoint_path: Path
    history_csv_path: Path


@dataclass
class TrainingContext:
    """Mutable and immutable coordinates for an active training run session.

    Constructed at session initialization to provide deterministic access across
    epoch loops, validation passes, hardware sentinels, and rollback managers.
    """

    model_name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    criterion: Any
    train_loader: DataLoader
    val_loader: DataLoader | None
    device_info: DeviceInfo
    policy: ExecutionPolicy
    sentinel: SentinelGuard
    governor: SmartTrainingGovernor
    sota_tracker: SotaTracker
    vault: MetricVault
    paths: TrainingPaths
    config: dict[str, Any]
    model_info: dict[str, Any]
    args: Any
    total_epochs: int
    accumulation_steps: int = 1
    scaler: Any | None = None
    resume_state: ResumeState | None = None
    raw_model: torch.nn.Module | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
