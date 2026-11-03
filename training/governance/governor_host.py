"""Shared attribute surface of ``SmartTrainingGovernor`` and its mixins.

The governor is assembled from ``EpochAuditMixin``, ``PropulsionMixin`` and
``GovernorStateMixin``. Each mixin reads attributes that the governor sets in
``__init__``; ``GovernorHost`` declares them once so every mixin sees the same
types. The governor's only instance attributes are the grouped states (annotations
here); every historical flat name is a ``Delegated`` descriptor onto one of them,
and the thin sub-state forwarders live here too. Code inside the governor writes
through the groups; callers may keep using the flat names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from training.governance.curriculum import CurriculumState
from training.governance.delegation import Delegated
from training.governance.governor_groups import GovernorRuntime, GovernorSpec, GovernorTuning
from training.governance.metrics import MetricRegistry
from training.governance.sota import HistoryEntry, SotaTracker
from training.governance.thermal import ThermalState

_LOCKS = "tuning.locks"
_RANK = "runtime.rank"
_WEIGHTS = "runtime.weights"
_CALIBRATION = "runtime.calibration"
_LR = "runtime.lr"
_COUNTERS = "runtime.counters"
_FOCUS = "runtime.focus"


class GovernorHost:
    """Attributes the governor provides to its mixins."""

    # Grouped state (the only instance attributes of the governor).
    spec: GovernorSpec
    tuning: GovernorTuning
    runtime: GovernorRuntime
    metrics: MetricRegistry
    curriculum: CurriculumState
    thermal: ThermalState
    sota: SotaTracker

    # Inputs the governor was built from.
    model_info: Delegated[dict[str, Any]] = Delegated("spec")
    config: Delegated[dict[str, Any]] = Delegated("spec")
    stab: Delegated[dict[str, Any]] = Delegated("spec")
    enabled: Delegated[bool] = Delegated("spec")
    task_type: Delegated[str] = Delegated("spec")
    sota_targets: Delegated[dict[str, float]] = Delegated("spec")

    # Tuning knobs read from the model's optimization block and the config.
    manifold_maturity: Delegated[int] = Delegated("tuning")
    plateau_priority: Delegated[str] = Delegated("tuning")
    cooling_factor: Delegated[float] = Delegated("tuning")
    clamp_range: Delegated[list[float]] = Delegated("tuning")
    sharpening_rate: Delegated[float] = Delegated("tuning")
    min_delta: Delegated[float] = Delegated("tuning")
    jolt_cooldown: Delegated[int] = Delegated(_LOCKS)
    stabilization_lock: Delegated[int] = Delegated(_LOCKS)
    breakout_threshold: Delegated[float] = Delegated(_LOCKS)

    # Loss-shaping state.
    min_temp: Delegated[float] = Delegated(_CALIBRATION)
    current_clamp: Delegated[float] = Delegated(_CALIBRATION)
    thermal_floor: Delegated[dict[str, float]] = Delegated(_CALIBRATION)
    current_rank_weight: Delegated[float] = Delegated(_RANK)
    current_rank_margin: Delegated[float] = Delegated(_RANK)
    max_rank_weight: Delegated[float] = Delegated(_RANK)
    min_rank_margin: Delegated[float] = Delegated(_RANK)
    current_spearman_weight: Delegated[float] = Delegated(_RANK)
    current_emd_weight: Delegated[float] = Delegated(_WEIGHTS)
    current_ssim_weight: Delegated[float] = Delegated(_WEIGHTS)
    current_huber_delta: Delegated[float] = Delegated(_WEIGHTS)
    current_conf_gate_str: Delegated[float] = Delegated(_WEIGHTS)
    current_lpips_weight: Delegated[float] = Delegated(_WEIGHTS)
    current_mag_weight: Delegated[float] = Delegated(_WEIGHTS)
    current_dir_weight: Delegated[float] = Delegated(_WEIGHTS)

    # Per-epoch counters and windows.
    metric_focus_epochs_remaining: Delegated[int] = Delegated(_FOCUS)
    metric_focus_target: Delegated[str | None] = Delegated(_FOCUS)
    metric_focus_last_fired: Delegated[dict[str, int]] = Delegated(_FOCUS)
    lr_multiplier: Delegated[float] = Delegated(_LR)
    head_lr_multiplier: Delegated[float] = Delegated(_LR)
    jolt_window_remaining: Delegated[int] = Delegated(_LR)
    trigger_mini_swa: Delegated[bool] = Delegated(_LR)
    last_action_epoch: Delegated[int] = Delegated(_COUNTERS)
    epoch_count: Delegated[int] = Delegated(_COUNTERS)
    session_epoch_count: Delegated[int] = Delegated(_COUNTERS)
    last_jolt_epoch: Delegated[int] = Delegated(_COUNTERS)
    spatial_lock_remaining: Delegated[int] = Delegated(_COUNTERS)
    last_res_jump_epoch: Delegated[int] = Delegated(_COUNTERS)

    # Monotonic Ratchet pair-rule commit; ``SmartTrainingGovernor._commit_rung`` is the
    # only implementation and the only writer of the (resolution, fraction) rung.
    _commit_rung: Callable[[int, float], None]

    # -------------------------------------------------------------------------
    # Delegated Properties
    # -------------------------------------------------------------------------
    current_fraction: Delegated[float] = Delegated("curriculum", cast=float)
    current_res: Delegated[int] = Delegated("curriculum")
    res_ladder: Delegated[list[int]] = Delegated("curriculum", cast=list)
    current_batch: Delegated[int] = Delegated("curriculum")
    current_acc: Delegated[int] = Delegated("curriculum")
    target_effective_batch: Delegated[int] = Delegated("curriculum")
    fraction_increment: Delegated[float] = Delegated("curriculum", cast=float)
    current_temp: Delegated[float] = Delegated("thermal", "temperature", cast=float)
    best_quality: Delegated[float] = Delegated("sota", cast=float)
    prev_quality: Delegated[float] = Delegated("sota", cast=float)
    prev_loss: Delegated[float] = Delegated("sota", cast=float)
    target_quality_score: Delegated[float] = Delegated("sota", cast=float)
    recovery_streak: Delegated[int] = Delegated("sota")
    consecutive_rollbacks: Delegated[int] = Delegated("sota")
    rollback_history: Delegated[dict[int, int]] = Delegated("sota", cast=dict)
    gate_relaxation_epochs: Delegated[int] = Delegated("sota")
    sota_resolution: Delegated[int | None] = Delegated("sota")
    breakout_lock: Delegated[int] = Delegated("sota")
    history: Delegated[list[HistoryEntry]] = Delegated("sota", cast=list)
    failure_log: Delegated[dict[str, int]] = Delegated("sota", cast=dict)
    stabilization_epochs: Delegated[int] = Delegated("sota")
    cooldown_remaining: Delegated[int] = Delegated("sota")
    current_stress: Delegated[float] = Delegated("sota", cast=float)
    max_stress_stuck_epochs: Delegated[int] = Delegated("sota")
    plateau_patience: Delegated[int] = Delegated("sota")
    loop_breaker_enabled: Delegated[bool] = Delegated("sota")
    loop_breaker_threshold: Delegated[int] = Delegated("sota")
    loop_breaker_strategy: Delegated[str] = Delegated("sota")

    def update_batch_and_accumulation(self, new_batch: int) -> None:
        """Dynamically update current batch size and recompute accumulation steps."""
        self.curriculum.update_batch_and_accumulation(new_batch)

    def reset_best(self) -> None:
        """Reset historical SOTA tracking."""
        self.sota.reset_best()

    def get_active_drift_gate(self, config_gate: float) -> float:
        """Retrieve active drift gate considering relaxation."""
        return self.sota.get_active_drift_gate(config_gate)

    def get_active_regression_limit(self, config_limit: int) -> int:
        """Retrieve active regression limit considering rollback history."""
        return self.sota.get_active_regression_limit(config_limit, self.current_res)
