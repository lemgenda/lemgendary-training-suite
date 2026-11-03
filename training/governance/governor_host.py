"""Shared attribute surface of ``SmartTrainingGovernor`` and its mixins.

The governor is assembled from ``EpochAuditMixin``, ``PropulsionMixin`` and
``GovernorStateMixin``. Each mixin reads attributes that the governor sets in
``__init__``; ``GovernorHost`` declares them once so every mixin sees the same
types. Plain attributes are annotations only (no runtime value); the flat views
onto the curriculum, SOTA and thermal sub-states are ``Delegated`` descriptors,
and the thin sub-state forwarders live here too.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from training.governance.curriculum import CurriculumState
from training.governance.delegation import Delegated
from training.governance.metrics import MetricRegistry
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState

# One ``SotaTracker.history`` row: (quality, validation loss, train loss).
HistoryEntry = tuple[float, float | None, float | None]


class GovernorHost:
    """Attributes the governor provides to its mixins."""

    # Inputs and sub-states.
    model_info: dict[str, Any]
    config: dict[str, Any]
    stab: dict[str, Any]
    enabled: bool
    metrics: MetricRegistry
    curriculum: CurriculumState
    thermal: ThermalState
    sota: SotaTracker

    # Tuning knobs read from the model's optimization block and the config.
    task_type: str
    manifold_maturity: int
    plateau_priority: str
    cooling_factor: float
    clamp_range: list[float]
    jolt_cooldown: int
    stabilization_lock: int
    breakout_threshold: float
    sharpening_rate: float
    min_delta: float
    sota_targets: dict[str, float]

    # Loss-shaping state.
    min_temp: float
    current_clamp: float
    current_rank_weight: float
    current_rank_margin: float
    max_rank_weight: float
    min_rank_margin: float
    current_spearman_weight: float
    current_emd_weight: float
    current_ssim_weight: float
    current_huber_delta: float
    current_conf_gate_str: float
    current_lpips_weight: float
    current_mag_weight: float
    current_dir_weight: float

    # Per-epoch counters and windows.
    metric_focus_epochs_remaining: int
    metric_focus_target: str | None
    metric_focus_last_fired: dict[str, int]
    thermal_floor: dict[str, float]
    lr_multiplier: float
    head_lr_multiplier: float
    jolt_window_remaining: int
    trigger_mini_swa: bool
    last_action_epoch: int
    epoch_count: int
    session_epoch_count: int
    last_jolt_epoch: int
    spatial_lock_remaining: int
    last_res_jump_epoch: int

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
