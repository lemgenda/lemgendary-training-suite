"""SOTA tracking, recovery streaks, rollback history, and drift-gated decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from training.governance.delegation import Delegated

logger = logging.getLogger("lemtrain.governance.sota")

# One ``SotaTracker.history`` row: (quality, validation loss, train loss).
HistoryEntry = tuple[float, float | None, float | None]


@dataclass
class QualityMemory:
    """Best and previous quality, the quality target and the recent-epoch history."""

    best_quality: float = 0.0
    prev_quality: float = 0.0
    prev_loss: float = 999.0
    target_quality_score: float = 1.0
    sota_targets: dict[str, float] = field(default_factory=dict)
    history: list[HistoryEntry] = field(default_factory=list)


@dataclass
class RollbackState:
    """Rollback counts, gate relaxation and the breakout lock of the loop breaker."""

    recovery_streak: int = 0
    consecutive_rollbacks: int = 0
    rollback_history: dict[int, int] = field(default_factory=dict)
    gate_relaxation_epochs: int = 0
    sota_resolution: int | None = None
    breakout_lock: int = 0


@dataclass
class StressState:
    """NPP failure log, stabilization and cooldown windows and the stress protocol level."""

    failure_log: dict[str, int] = field(default_factory=dict)
    stabilization_epochs: int = 0
    cooldown_remaining: int = 0
    current_stress: float = 0.0
    max_stress_stuck_epochs: int = 0


@dataclass
class LoopBreakerPolicy:
    """Loop-breaker switches, plateau patience and the task the tracker serves."""

    loop_breaker_enabled: bool = True
    loop_breaker_threshold: int = 2
    loop_breaker_strategy: str = "auto"
    plateau_patience: int = 6
    task_type: str = "quality"


_GROUP_TYPES: dict[str, type] = {
    "quality": QualityMemory,
    "rollback": RollbackState,
    "stress": StressState,
    "policy": LoopBreakerPolicy,
}
_FIELD_GROUP: dict[str, str] = {
    f.name: group for group, cls in _GROUP_TYPES.items() for f in fields(cls)
}


class SotaTracker:
    """Manages SOTA metric history, stability locks, and drift-gate policies.

    Used in two distinct roles across the training suite:
    1. Global SOTA tracker (``ctx.governance.sota_tracker`` in ``core_loop.py``): tracks the
       all-time best Quality Score across the entire run to gate ``best.pth`` saves.
    2. Per-rung curriculum tracker (``governor.sota`` in ``SmartTrainingGovernor``): tracks
       plateau patience, stabilization windows, and recent epoch history within the active
       curriculum rung.

    The state lives in four groups; every field stays reachable as a flat attribute and
    as a keyword of the constructor (``SotaTracker(best_quality=0.9)``).
    """

    best_quality: Delegated[float] = Delegated("quality")
    prev_quality: Delegated[float] = Delegated("quality")
    prev_loss: Delegated[float] = Delegated("quality")
    target_quality_score: Delegated[float] = Delegated("quality")
    sota_targets: Delegated[dict[str, float]] = Delegated("quality")
    history: Delegated[list[HistoryEntry]] = Delegated("quality")
    recovery_streak: Delegated[int] = Delegated("rollback")
    consecutive_rollbacks: Delegated[int] = Delegated("rollback")
    rollback_history: Delegated[dict[int, int]] = Delegated("rollback")
    gate_relaxation_epochs: Delegated[int] = Delegated("rollback")
    sota_resolution: Delegated[int | None] = Delegated("rollback")
    breakout_lock: Delegated[int] = Delegated("rollback")
    failure_log: Delegated[dict[str, int]] = Delegated("stress")
    stabilization_epochs: Delegated[int] = Delegated("stress")
    cooldown_remaining: Delegated[int] = Delegated("stress")
    current_stress: Delegated[float] = Delegated("stress")
    max_stress_stuck_epochs: Delegated[int] = Delegated("stress")
    loop_breaker_enabled: Delegated[bool] = Delegated("policy")
    loop_breaker_threshold: Delegated[int] = Delegated("policy")
    loop_breaker_strategy: Delegated[str] = Delegated("policy")
    plateau_patience: Delegated[int] = Delegated("policy")
    task_type: Delegated[str] = Delegated("policy")

    def __init__(self, **overrides: Any) -> None:
        self.quality = QualityMemory()
        self.rollback = RollbackState()
        self.stress = StressState()
        self.policy = LoopBreakerPolicy()
        for name, value in overrides.items():
            if name not in _FIELD_GROUP:
                raise TypeError(f"SotaTracker() got an unexpected keyword argument '{name}'")
            setattr(getattr(self, _FIELD_GROUP[name]), name, value)

    def record_epoch(
        self,
        current_quality: float,
        current_loss: float | None = None,
        train_loss: float | None = None,
    ) -> bool:
        """Record an epoch's metrics and update history buffer. Returns True if new SOTA."""
        self.history.append((current_quality, current_loss, train_loss))
        if len(self.history) > 5:
            self.history.pop(0)

        is_new_sota = (current_quality > self.best_quality) or (
            self.best_quality == 0.0 and current_quality > 0.0
        )
        if is_new_sota:
            self.stress.current_stress = min(self.current_stress, 0.0)
            self.quality.best_quality = current_quality

        return is_new_sota

    def reset_best(self) -> None:
        """Purge SOTA memory to establish fresh baseline for current manifold."""
        self.quality.best_quality = 0.0
        self.quality.prev_quality = 0.0
        self.quality.prev_loss = 999.0
        self.history.clear()
        self.stress.stabilization_epochs = 2
        logger.info("SOTA Memory Purged. Establishing fresh baseline for current manifold.")

    def get_active_drift_gate(self, config_gate: float) -> float:
        """Return dynamic drift gate with relaxation decay."""
        if self.gate_relaxation_epochs > 0:
            self.rollback.gate_relaxation_epochs -= 1
            return min(0.80, config_gate * 0.85)
        return config_gate

    def get_active_regression_limit(self, config_limit: int, current_res: int) -> int:
        """Return active regression epoch limit adjusted for rollback history."""
        if self.gate_relaxation_epochs > 0 or self.rollback_history.get(current_res, 0) > 0:
            return max(config_limit, self.plateau_patience + 2)
        return config_limit

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for checkpoint payload."""
        return {
            "best_quality": self.best_quality,
            "prev_quality": self.prev_quality,
            "prev_loss": self.prev_loss,
            "target_quality_score": self.target_quality_score,
            "sota_targets": dict(self.sota_targets),
            "recovery_streak": self.recovery_streak,
            "consecutive_rollbacks": self.consecutive_rollbacks,
            "rollback_history": {str(k): v for k, v in self.rollback_history.items()},
            "gate_relaxation_epochs": self.gate_relaxation_epochs,
            "sota_resolution": self.sota_resolution,
            "breakout_lock": self.breakout_lock,
            "history": list(self.history),
            "failure_log": dict(self.failure_log),
            "stabilization_epochs": self.stabilization_epochs,
            "cooldown_remaining": self.cooldown_remaining,
            "stress": self.current_stress,
            "max_stress_stuck_epochs": self.max_stress_stuck_epochs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SotaTracker:
        """Deserialize state from dictionary."""
        raw_rh = data.get("rollback_history", {})
        rh = {int(k): int(v) for k, v in raw_rh.items()}
        return cls(
            best_quality=float(data.get("best_quality", 0.0)),
            prev_quality=float(data.get("prev_quality", 0.0)),
            prev_loss=float(data.get("prev_loss", 999.0)),
            target_quality_score=float(data.get("target_quality_score", 1.0)),
            sota_targets=dict(data.get("sota_targets", {})),
            recovery_streak=int(data.get("recovery_streak", 0)),
            consecutive_rollbacks=int(data.get("consecutive_rollbacks", 0)),
            rollback_history=rh,
            gate_relaxation_epochs=int(data.get("gate_relaxation_epochs", 0)),
            sota_resolution=data.get("sota_resolution"),
            breakout_lock=int(data.get("breakout_lock", 0)),
            history=list(data.get("history", [])),
            failure_log=dict(data.get("failure_log", {})),
            stabilization_epochs=int(data.get("stabilization_epochs", 0)),
            cooldown_remaining=int(data.get("cooldown_remaining", 0)),
            current_stress=float(data.get("stress", 0.0)),
            max_stress_stuck_epochs=int(data.get("max_stress_stuck_epochs", 0)),
        )
