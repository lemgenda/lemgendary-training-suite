"""SOTA tracking, recovery streaks, rollback history, and drift-gated decisions."""

from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger("lemtrain.governance.sota")


@dataclass
class SotaTracker:
    """Manages SOTA metric history, stability locks, and rollback breakout policies."""

    best_quality: float = 0.0
    prev_quality: float = 0.0
    prev_loss: float = 999.0
    target_quality_score: float = 1.0
    sota_targets: dict[str, float] = field(default_factory=dict)
    recovery_streak: int = 0
    consecutive_rollbacks: int = 0
    rollback_history: dict[int, int] = field(default_factory=dict)
    gate_relaxation_epochs: int = 0
    sota_resolution: int | None = None
    breakout_lock: int = 0
    history: list[tuple[float, float | None, float | None]] = field(default_factory=list)
    failure_log: dict[str, int] = field(default_factory=dict)
    stabilization_epochs: int = 0
    cooldown_remaining: int = 0
    current_stress: float = 0.0
    max_stress_stuck_epochs: int = 0
    loop_breaker_enabled: bool = True
    loop_breaker_threshold: int = 2
    loop_breaker_strategy: str = "auto"
    plateau_patience: int = 6
    task_type: str = "quality"

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

        is_new_sota = (current_quality > self.best_quality) or (self.best_quality == 0.0 and current_quality > 0.0)
        if is_new_sota:
            if self.current_stress > 0.0:
                self.current_stress = 0.0
            self.best_quality = current_quality

        return is_new_sota

    def reset_best(self) -> None:
        """Purge SOTA memory to establish fresh baseline for current manifold."""
        self.best_quality = 0.0
        self.prev_quality = 0.0
        self.prev_loss = 999.0
        self.history.clear()
        self.stabilization_epochs = 2
        logger.info("SOTA Memory Purged. Establishing fresh baseline for current manifold.")

    def register_rollback(
        self,
        current_res: int,
        res_ladder: list[int],
    ) -> dict[str, Any]:
        """Record a rollback occurrence and trigger breakout or gate relaxation if threshold is met."""
        result: dict[str, Any] = {
            "breakout_triggered": False,
            "strategy": None,
            "new_res": None,
            "reset_best": False,
        }
        if not self.loop_breaker_enabled:
            return result

        self.consecutive_rollbacks += 1
        res_key = current_res
        self.rollback_history[res_key] = self.rollback_history.get(res_key, 0) + 1

        if self.rollback_history.get(res_key, 0) >= self.loop_breaker_threshold:
            strategy = self.loop_breaker_strategy
            if strategy == "auto":
                strategy = "escalate" if self.task_type == "quality" else "relax"
            result["strategy"] = strategy

            current_idx = res_ladder.index(current_res) if current_res in res_ladder else -1
            has_higher_res = 0 <= current_idx < len(res_ladder) - 1
            target_res = (
                self.sota_resolution
                if (self.sota_resolution is not None and self.sota_resolution > current_res)
                else (res_ladder[current_idx + 1] if has_higher_res else None)
            )

            if strategy == "escalate" and target_res is not None:
                self.rollback_history[target_res] = 0
                self.consecutive_rollbacks = 0
                self.gate_relaxation_epochs = 0
                self.breakout_lock = 8
                self.reset_best()
                result["breakout_triggered"] = True
                result["new_res"] = target_res
                result["reset_best"] = True
                logger.warning(
                    "Resolution-Regression Lock detected at %dpx! Promoting to %dpx with breakout protection.",
                    current_res,
                    target_res,
                )
            elif strategy in ["escalate", "relax"]:
                self.gate_relaxation_epochs = 6
                self.consecutive_rollbacks = 0
                result["breakout_triggered"] = True
                logger.warning(
                    "Stagnation Rollback Lock detected at %dpx! Dynamic Gate Relaxation active for 6 epochs.",
                    res_key,
                )

        return result

    def get_active_drift_gate(self, config_gate: float) -> float:
        """Return dynamic drift gate with relaxation decay."""
        if self.gate_relaxation_epochs > 0:
            self.gate_relaxation_epochs -= 1
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
    def from_dict(cls, data: dict[str, Any]) -> "SotaTracker":
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
