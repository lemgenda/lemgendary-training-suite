"""Stability signals and per-epoch counters of the smart training governor."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("lemtrain.governance.governor")

AuditResult = tuple[bool, bool, bool, bool, bool, bool, bool, str]

# The six change flags plus the early-stop flag of an audit that changed nothing.
NO_CHANGE = (False, False, False, False, False, False, False)

# Loss-weight overrides applied while a metric-focus window is open. Adding a focus target
# is a new table row; the tick itself never changes.
METRIC_FOCUS_WEIGHTS: dict[str, dict[str, float]] = {
    "srcc": {"current_spearman_weight": 3.0, "current_rank_weight": 2.0},
    "plcc": {
        "current_emd_weight": 1.5,
        "current_rank_weight": 0.3,
        "current_spearman_weight": 0.2,
    },
    "ssim": {"current_ssim_weight": 0.15},
    "lpips": {"current_lpips_weight": 0.08},
    "fid": {"current_stress": 5.0},
    "mae": {"current_huber_delta": 0.1},
    "dir_acc": {"current_dir_weight": 2.0},
    "win_rate": {"current_dir_weight": 2.0, "current_conf_gate_str": 1.5},
    "profit_factor": {"current_dir_weight": 2.0, "current_mag_weight": 0.2},
    "max_drawdown": {"current_mag_weight": 0.1, "current_conf_gate_str": 2.0},
    "tp_mae": {"current_mag_weight": 1.2},
    "sl_mae": {"current_mag_weight": 1.2},
}


@dataclass
class AuditDirectives:
    """Mutable change flags and status fragments collected during one epoch audit."""

    f_changed: bool = False
    r_changed: bool = False
    lr_changed: bool = False
    t_changed: bool = False
    c_changed: bool = False
    b_changed: bool = False
    msg_parts: list[str] = field(default_factory=list)

    def flags(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Return the six change flags in ``audit_epoch`` result order."""
        return (
            self.f_changed,
            self.r_changed,
            self.lr_changed,
            self.t_changed,
            self.c_changed,
            self.b_changed,
        )


class EpochSignalsMixin:
    """Counters, holds and stability detectors used by ``EpochAuditMixin.audit_epoch``."""

    # Class-level defaults for the counters this mixin advances; the governor
    # initialises the instance values (``breakout_lock`` is a ``Delegated`` there).
    epoch_count: int = 0
    session_epoch_count: int = 0
    metric_focus_epochs_remaining: int = 0
    breakout_lock: int = 0

    def advance_counters(self) -> None:
        """Count the epoch, tick an open metric-focus window and the breakout lock."""
        self.epoch_count += 1
        self.session_epoch_count += 1

        if self.metric_focus_epochs_remaining > 0:
            self.tick_metric_focus()

        if self.breakout_lock > 0:
            self.breakout_lock -= 1

    def tick_metric_focus(self) -> None:
        """Apply the active metric-focus loss weights and count the focus window down."""
        self.metric_focus_epochs_remaining -= 1
        for name, value in METRIC_FOCUS_WEIGHTS.get(self.metric_focus_target, {}).items():
            setattr(self, name, value)

        self.lr_multiplier = self.cooling_factor
        self.head_lr_multiplier = min(self.lr_multiplier * 2.0, 1.5)
        self.jolt_window_remaining = 5

        if self.metric_focus_epochs_remaining == 0:
            self.metric_focus_target = None

    def track_recovery_streak(self, sentinel_trigger_rate: float) -> None:
        """Count stress-free epochs and break the stabilization lock after two."""
        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if (
                self.recovery_streak >= 2
                and self.stabilization_epochs > 0
                and self.spatial_lock_remaining == 0
            ):
                self.stabilization_epochs = 0
                logger.info("[LAUNCH] [NPP] Stress at zero. Breaking stabilization lock.")
        else:
            self.recovery_streak = 0

    def stabilization_hold(
        self, current_quality: float, current_loss: float | None, msg_parts: list[str]
    ) -> str | None:
        """Hold the manifold while stabilization is active; ``None`` if the shield broke."""
        if (
            current_quality < self.best_quality * (1.0 - self.breakout_threshold)
            and self.best_quality > 0
        ):
            msg_parts.append(
                "[BREAKOUT] Shield shattered! Quality dropped "
                f"{(1 - current_quality / self.best_quality) * 100:.1f}%."
            )
            self.stabilization_epochs = 0
            return None

        self.stabilization_epochs -= 1
        recovery_delta = self.min_delta if self.task_type != "quality" else 0.05
        if current_quality - self.prev_quality > recovery_delta:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
            self.stabilization_epochs = max(0, self.stabilization_epochs - 1)
            msg_parts.append("[RECOVERY] Rapid quality gain detected. Cooldown and Lock shortened.")

        self.remember_epoch(current_quality, current_loss)
        status_msg = (
            f"[SIGNAL] Anchoring Manifold... (Cooldown: {self.cooldown_remaining})"
            if self.cooldown_remaining > 0
            else "[SIGNAL] Anchoring Manifold..."
        )
        if msg_parts:
            status_msg = " | ".join(msg_parts) + " | " + status_msg
        return status_msg

    def remember_epoch(self, current_quality: float, current_loss: float | None) -> None:
        """Store this epoch's quality, and its loss when non-zero, as the previous values."""
        self.prev_quality = current_quality
        if current_loss:
            self.prev_loss = current_loss

    def detect_overfitting(self) -> bool:
        """Return True when train loss falls while validation loss rises over 3 epochs."""
        if len(self.history) >= 3:
            train_losses = [h[2] for h in self.history if len(h) > 2 and h[2] is not None]
            val_losses = [h[1] for h in self.history if h[1] is not None]
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                train_trend = train_losses[-1] - train_losses[-3]
                val_trend = val_losses[-1] - val_losses[-3]
                if train_trend < -1e-4 and val_trend > 1e-4:
                    return True
        return False

    def detect_turbulence(self) -> bool:
        """Return True when the last three quality deltas alternate in sign."""
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i - 1] for i in range(1, len(q_values))]
            if all(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas))) and all(
                abs(d) > self.min_delta * 2 for d in deltas
            ):
                return True
        return False

    def apply_retreat_or_cooling(
        self,
        d: AuditDirectives,
        should_retreat: bool,
        sentinel_trigger_rate: float,
        current_state: tuple[int, float],
    ) -> None:
        """Record an NPP failure and cool LR, or cool temperature under sentinel stress."""
        if should_retreat:
            self.failure_log[str(current_state)] = self.failure_log.get(str(current_state), 0) + 1
            d.msg_parts.append(
                f"[WARNING] NPP FAILURE: State {current_state} (Count: "
                f"{self.failure_log[str(current_state)]})"
            )

            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                d.c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                d.t_changed = True
                d.msg_parts.append("NPP LOOP: Forcing Numerical Shakeup")

            # Monotonic ratchet: resolution and data fraction never retreat in production;
            # recovery is LR cooling only.
            if not d.r_changed:
                self.lr_multiplier = self.cooling_factor
                d.lr_changed = True
                self.stabilization_epochs = self.stabilization_lock
                self.cooldown_remaining = 5
                d.msg_parts.append(
                    f"RECOIL: Retaining data fraction at {self.current_fraction * 100:.0f}% | "
                    "Cooling LR to stabilize manifold"
                )

        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            self.current_stress = max(0.0, self.current_stress - 0.25)
            d.lr_changed = True
            d.t_changed = True
            self.stabilization_epochs = 2
            d.msg_parts.append(
                f"COOLING: Stress {sentinel_trigger_rate * 100:.1f}% -> Temp "
                f"{self.current_temp:.2f} | DataStress {self.current_stress:.1f}"
            )

    def tick_jolt_window(
        self, d: AuditDirectives, delta_q: float, should_retreat: bool, is_regressing: bool
    ) -> None:
        """Sustain or abort an open jolt window."""
        self.jolt_window_remaining -= 1
        collapse_threshold = (
            (self.prev_quality * -0.03)
            if (self.prev_quality and self.prev_quality > 1.0)
            else -0.015
        )
        if delta_q < collapse_threshold or should_retreat or is_regressing:
            self.jolt_window_remaining = 0
            self.lr_multiplier = self.cooling_factor
            self.head_lr_multiplier = self.cooling_factor
            d.lr_changed = True
            d.msg_parts.append(
                f"[JOLT SHIELD] Early collapse triggered (Regression: {delta_q:.4f} < "
                f"{collapse_threshold:.4f}). Cooling LR."
            )
        else:
            d.lr_changed = True
            d.msg_parts.append(
                f"SUSTAINED JOLT: Window Active ({self.jolt_window_remaining} epochs remaining "
                f"| Head LR: {self.head_lr_multiplier:.2f}x | Backbone LR: "
                f"{self.lr_multiplier:.2f}x)"
            )
