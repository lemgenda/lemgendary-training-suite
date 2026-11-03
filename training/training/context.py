"""Immutable Execution Context for LemGendary Training Suite.

Encapsulates all runtime components: hardware policy, models, optimizers,
schedulers, data loaders, governance mechanisms, and checkpoint managers.
Components are grouped into small cohesive dataclasses (model stack, data,
hardware, governance, run settings, mutable run state) hung off
:class:`TrainingContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from training.checkpoint import CheckpointLifecycleManager, MetricVault, ResumeState
from training.governance.governor import SmartTrainingGovernor
from training.governance.sota import SotaTracker
from training.hardware.discovery import DeviceInfo
from training.hardware.policy import ExecutionPolicy
from training.hardware.sentinel import SentinelGuard

if TYPE_CHECKING:
    from training.training.swa import SwaPhase


@dataclass(frozen=True)
class ModelsHubPaths:
    """The run's ``LemGendaryModels/<model_key>`` artefact locations."""

    root: Path
    checkpoint_dir: Path
    progress_path: Path
    best_path: Path
    metrics_csv: Path


@dataclass(frozen=True)
class TrainingPaths:
    """Filesystem pathways for checkpoints, telemetry, exports, and Hub synchronizations."""

    project_root: Path
    local_checkpoint_dir: Path
    export_dir: Path
    progress_local_path: Path
    best_checkpoint_path: Path
    history_csv_path: Path
    models_hub: ModelsHubPaths | None = None


@dataclass
class ModelStack:
    """The network, its optimisation machinery and parallel wrapper."""

    model: Any
    optimizer: torch.optim.Optimizer
    scheduler: Any
    criterion: Any
    scaler: Any | None = None
    raw_model: torch.nn.Module | None = None
    parallel_strategy: Any | None = None


@dataclass
class DataComponents:
    """Train/validation loaders and the gradient accumulation they are consumed with."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    accumulation_steps: int = 1


@dataclass
class HardwareComponents:
    """Resolved device, execution policy and hardware sentinel."""

    device_info: DeviceInfo
    policy: ExecutionPolicy
    sentinel: SentinelGuard


@dataclass
class GovernanceComponents:
    """Curriculum governor, SOTA tracking, metric vault, checkpoint lifecycle and SWA."""

    governor: SmartTrainingGovernor
    sota_tracker: SotaTracker
    vault: MetricVault
    lifecycle_manager: CheckpointLifecycleManager | None = None
    # Stochastic Weight Averaging component; None when SWA is disabled.
    swa: SwaPhase | None = None


@dataclass
class RunSettings:
    """Static run configuration: paths, configs, CLI args and the epoch budget."""

    paths: TrainingPaths
    config: dict[str, Any]
    model_info: dict[str, Any]
    args: Any
    total_epochs: int


@dataclass
class ResumeGuards:
    """One-shot guards so an intra-epoch resume fast-forward is applied only once."""

    train_step_handled: bool = False
    val_handled: bool = False
    val_step_handled: bool = False


@dataclass
class RunState:
    """Mutable per-run state: resume source, cached metrics and resume guards."""

    resume_state: ResumeState | None = None
    last_train_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    guards: ResumeGuards = field(default_factory=ResumeGuards)


@dataclass
class TrainingContext:
    """Mutable and immutable coordinates for an active training run session.

    Constructed at session initialization to provide deterministic access across
    epoch loops, validation passes, hardware sentinels, and rollback managers.
    """

    model_name: str
    models: ModelStack
    data: DataComponents
    hardware: HardwareComponents
    governance: GovernanceComponents
    run: RunSettings
    state: RunState = field(default_factory=RunState)


def detached_copy(ctx: TrainingContext) -> TrainingContext:
    """Return a copy of ``ctx`` whose component groups can be reassigned independently.

    Equivalent to a shallow ``dataclasses.replace`` of the former flat context: every
    group (and the resume guards) is a fresh shallow copy, while the components they
    hold (model, loaders, dicts, ...) stay shared.
    """
    return replace(
        ctx,
        models=replace(ctx.models),
        data=replace(ctx.data),
        hardware=replace(ctx.hardware),
        governance=replace(ctx.governance),
        run=replace(ctx.run),
        state=replace(ctx.state, guards=replace(ctx.state.guards)),
    )
