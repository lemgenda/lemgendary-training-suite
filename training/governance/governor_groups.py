"""Attribute groups of ``SmartTrainingGovernor``.

The governor keeps its configuration and per-epoch state in these small dataclasses;
``GovernorHost`` exposes every field under its historical flat name through
``Delegated`` descriptors, so callers and checkpoints see the same attributes as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GovernorSpec:
    """Inputs the governor was built from, plus the derived task type and SOTA targets."""

    model_info: dict[str, Any]
    config: dict[str, Any]
    stab: dict[str, Any]
    enabled: bool = True
    task_type: str = "quality"
    sota_targets: dict[str, float] = field(default_factory=dict)


@dataclass
class LockTuning:
    """Jolt cooldown, stabilization lock length and the emergency breakout threshold."""

    jolt_cooldown: int = 5
    stabilization_lock: int = 3
    breakout_threshold: float = 0.10


@dataclass
class GovernorTuning:
    """Tuning knobs read from the model's optimization block and the governor config."""

    manifold_maturity: int = 3
    plateau_priority: str = "data"
    cooling_factor: float = 0.5
    clamp_range: list[float] = field(default_factory=lambda: [15.0, 45.0])
    sharpening_rate: float = 0.98
    min_delta: float = 0.0005
    locks: LockTuning = field(default_factory=LockTuning)


@dataclass
class RankShaping:
    """Rank-loss weight and margin with their bounds, and the soft Spearman weight."""

    current_rank_weight: float = 0.8
    current_rank_margin: float = 0.10
    max_rank_weight: float = 1.5
    min_rank_margin: float = 0.05
    current_spearman_weight: float = 0.5

    @classmethod
    def from_stabilizers(cls, stab: dict[str, Any]) -> RankShaping:
        """Read the rank-shaping values from the stabilizer block."""
        return cls(
            current_rank_weight=float(stab.get("rank_weight", 0.8)),
            current_rank_margin=float(stab.get("rank_margin", 0.10)),
            max_rank_weight=float(stab.get("max_rank_weight", 1.5)),
            min_rank_margin=float(stab.get("min_rank_margin", 0.05)),
            current_spearman_weight=float(stab.get("soft_spearman_weight", 0.5)),
        )


@dataclass
class LossWeights:
    """Auxiliary loss weights the governor tunes during training."""

    current_emd_weight: float = 1.0
    current_ssim_weight: float = 0.0
    current_huber_delta: float = 1.0
    current_conf_gate_str: float = 1.0
    current_lpips_weight: float = 0.025
    current_mag_weight: float = 0.5
    current_dir_weight: float = 1.0

    @classmethod
    def from_stabilizers(cls, stab: dict[str, Any]) -> LossWeights:
        """Read the loss weights from the stabilizer block."""
        return cls(
            current_emd_weight=float(stab.get("emd_weight", 1.0)),
            current_ssim_weight=float(stab.get("ssim_weight", 0.0)),
            current_huber_delta=float(stab.get("huber_delta", 1.0)),
            current_conf_gate_str=float(stab.get("conf_gate_strength", 1.0)),
            current_lpips_weight=float(stab.get("lpips_weight", 0.025)),
            current_mag_weight=float(stab.get("mag_weight", 0.5)),
            current_dir_weight=float(stab.get("dir_weight", 1.0)),
        )


@dataclass
class Calibration:
    """Temperature floor, logit clamp and the per-rung thermal floors."""

    min_temp: float = 0.1
    current_clamp: float = 15.0
    thermal_floor: dict[str, float] = field(default_factory=dict)


@dataclass
class LrPropulsion:
    """Backbone and head LR multipliers, the jolt window and the mini-SWA pulse flag."""

    lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.0
    jolt_window_remaining: int = 0
    trigger_mini_swa: bool = False


@dataclass
class EpochCounters:
    """Epoch counts and the epochs of the last action, jolt and resolution jump."""

    epoch_count: int = 0
    session_epoch_count: int = 0
    last_action_epoch: int = 0
    last_jolt_epoch: int = -10
    last_res_jump_epoch: int = 0
    spatial_lock_remaining: int = 0


@dataclass
class MetricFocus:
    """Open metric-focus window: epochs left, target metric and last firing epochs."""

    metric_focus_epochs_remaining: int = 0
    metric_focus_target: str | None = None
    metric_focus_last_fired: dict[str, int] = field(default_factory=dict)


@dataclass
class GovernorRuntime:
    """Mutable per-epoch state of the governor, one group per concern."""

    rank: RankShaping = field(default_factory=RankShaping)
    weights: LossWeights = field(default_factory=LossWeights)
    calibration: Calibration = field(default_factory=Calibration)
    lr: LrPropulsion = field(default_factory=LrPropulsion)
    counters: EpochCounters = field(default_factory=EpochCounters)
    focus: MetricFocus = field(default_factory=MetricFocus)
