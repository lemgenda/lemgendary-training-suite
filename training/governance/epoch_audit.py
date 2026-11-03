"""Per-epoch audit of the smart training governor (signal detection and holds)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("lemtrain.governance.governor")

AuditResult = tuple[bool, bool, bool, bool, bool, bool, bool, str]


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


class EpochAuditMixin:
    """Epoch audit entry point and stability signals for ``SmartTrainingGovernor``."""

    # Class-level defaults for the counters this mixin advances; the governor
    # initialises the instance values (``breakout_lock`` is a ``Delegated`` there).
    epoch_count: int = 0
    session_epoch_count: int = 0
    metric_focus_epochs_remaining: int = 0
    breakout_lock: int = 0

    def get_phase(self) -> str:
        """Determine current training curriculum phase."""
        if self.curriculum.current_fraction < 1.0:
            return "FOUNDATION" if self.curriculum.current_fraction <= 0.40 else "EXPANSION"

        if (
            self.curriculum.res_ladder
            and self.curriculum.current_res < self.curriculum.res_ladder[-1]
        ):
            return "DEEPENING"

        return "REFINEMENT"

    def audit_epoch(
        self,
        current_quality: float,
        epochs_no_improve: int,
        sentinel_trigger_rate: float = 0.0,
        current_loss: float | None = None,
        plcc: float = 0.0,
        srcc: float = 0.0,
        target_std: float | None = None,
        force_jump: bool = False,
        train_loss: float | None = None,
        metrics_dict: dict[str, float] | None = None,
    ) -> tuple[bool, bool, bool, bool, bool, bool, bool, str]:
        """Audit single epoch performance and return optimization directives.

        Returns:
            Tuple of (f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed,
            early_stop_triggered, status_msg)
        """
        if not self.enabled and not force_jump:
            return False, False, False, False, False, False, False, ""

        self.epoch_count += 1
        self.session_epoch_count += 1

        if self.metric_focus_epochs_remaining > 0:
            self._tick_metric_focus()

        if self.breakout_lock > 0:
            self.breakout_lock -= 1

        if force_jump:
            return self._force_jump()

        self._track_recovery_streak(sentinel_trigger_rate)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.sota.record_epoch(current_quality, current_loss, train_loss)

        d = AuditDirectives()
        if self.stabilization_epochs > 0:
            held = self._stabilization_hold(current_quality, current_loss, d.msg_parts)
            if held is not None:
                return False, False, False, False, False, False, False, held

        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        is_regressing_shock = current_quality < self.best_quality * 0.95 and self.best_quality > 0
        if is_resuming and is_regressing_shock:
            self.prev_quality = current_quality
            if current_loss:
                self.prev_loss = current_loss
            return (
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                "[GUARD] [SHIELD] Resumption Shield Active. Buffering Momentum Shock.",
            )

        is_overfitting = self._detect_overfitting()

        self.lr_multiplier = 1.0
        phase = self.get_phase()

        delta_q = current_quality - self.prev_quality
        is_turbulent = self._detect_turbulence()

        is_plateaued = epochs_no_improve >= self.plateau_patience
        is_flat_leg = abs(delta_q) < self.min_delta and len(self.history) >= 2
        is_flat = is_plateaued or is_flat_leg

        abs_floor = 40.0 if self.task_type == "quality" else 0.0
        fidelity_floor = max(abs_floor, self.best_quality * 0.8)
        is_trapped = current_quality < fidelity_floor and len(self.history) >= 4
        if is_trapped:
            effective_min_delta = self.min_delta * 4
            is_flat_leg = abs(delta_q) < effective_min_delta
            is_flat = is_plateaued or is_flat_leg
            if is_flat:
                d.msg_parts.append("[TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        regress_threshold = (
            -0.03 if self.task_type in ["quality", "parameter_prediction"] else -0.01
        )
        is_regressing = (
            delta_q < (self.prev_quality * regress_threshold) if self.prev_quality else False
        )

        if self.task_type == "forex":
            is_collapsed = current_quality < 45.0
        else:
            is_collapsed = (current_quality < 0.05) or (plcc < -0.1)

        loss_is_stable = (
            (current_loss <= self.prev_loss * 1.05) if current_loss and self.prev_loss else True
        )
        is_expanding = phase in ["FOUNDATION", "EXPANSION"]

        if plcc < -0.01 and self.current_temp > 0.7 and (target_std is None or target_std >= 0.15):
            self.current_temp = 0.5
            self.current_clamp = 20.0
            d.t_changed = d.c_changed = True
            d.msg_parts.append("[THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

        if self.spatial_lock_remaining > 0:
            self.spatial_lock_remaining -= 1
            loss_is_exploding = (
                (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
            )
            if is_regressing and not loss_is_exploding:
                is_regressing = False
                d.msg_parts.append(
                    f"[SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})"
                )

        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(str(current_state), 0)

        if self.task_type in ["quality", "forex"]:
            is_turbulent = False
            if is_expanding:
                d.msg_parts.append(
                    "[SIGNAL] [RESONANCE] Turbulence detected but shielded. Holding manifold."
                )

        should_retreat = is_regressing or is_turbulent or is_collapsed
        if is_expanding and loss_is_stable and not is_collapsed:
            should_retreat = False
            if is_regressing:
                d.msg_parts.append(
                    "[GUARD] [MOMENTUM] Jitter detected but Loss is stable. Holding manifold."
                )

        if is_overfitting and is_expanding:
            should_retreat = False
            self.cooldown_remaining = 0
            d.msg_parts.append(
                "[RESCUE] [OVERFITTING] Overfitting detected. Forcing dataset expansion to "
                "introduce variety."
            )
        elif is_overfitting and self.current_fraction >= 1.0:
            should_retreat = False
            self.cooldown_remaining = 0
            if self.task_type == "forex":
                self.current_stress = min(2.0, self.current_stress + 0.5)
                d.msg_parts.append(
                    "[REGULARIZE] [OVERFITTING] Temporal fold divergence detected. Applying mild "
                    f"stabilization (Level {self.current_stress})."
                )
            else:
                self.current_stress = min(5.0, self.current_stress + 1.0)
                d.msg_parts.append(
                    "[RESCUE] [OVERFITTING] Dataset exhausted. Deploying Stress Protocol (Level "
                    f"{self.current_stress})."
                )

        self._apply_retreat_or_cooling(
            d, should_retreat, sentinel_trigger_rate, current_state, failures
        )

        stride_threshold = (
            0.75 if (self.current_res is not None and self.current_res < 512) else 0.90
        )
        if self.target_quality_score > 1.0:
            stride_threshold = stride_threshold * self.target_quality_score
        propulsion_allowed = not should_retreat and self.cooldown_remaining == 0
        not_regressing = delta_q >= -self.min_delta

        epochs_since_step = self.epoch_count - max(self.last_res_jump_epoch, self.last_action_epoch)
        min_soak = max(3, self.manifold_maturity)
        fraction_soak_complete = is_expanding and epochs_since_step >= min_soak
        deepening_ready = (
            phase == "DEEPENING"
            and current_quality >= stride_threshold
            and epochs_since_step >= min_soak
        )

        trigger_propulsion = propulsion_allowed and (
            (is_overfitting and is_expanding)
            or (is_plateaued and (fraction_soak_complete or not is_expanding))
            or (
                not_regressing
                and (
                    (is_flat_leg and not is_expanding) or fraction_soak_complete or deepening_ready
                )
            )
        )

        if self.jolt_window_remaining > 0:
            self._tick_jolt_window(d, delta_q, should_retreat, is_regressing)

        if trigger_propulsion:
            self.propel(d, phase, current_quality, is_flat, is_trapped, metrics_dict, plcc, srcc)

        if (
            not (is_regressing or is_turbulent or sentinel_trigger_rate > 0.15)
            and self.current_temp > self.min_temp
        ):
            self.sharpen(d, phase, current_state)

        self.prev_quality = current_quality
        if current_loss:
            self.prev_loss = current_loss

        if self.task_type == "quality":
            if self.cooldown_remaining == 0:
                self.current_temp = min(1.0, self.current_temp)
            self.current_clamp = min(self.clamp_range[1], self.current_clamp)

        early_stop_triggered = False
        if (
            epochs_no_improve >= self.plateau_patience
            and self.metric_focus_epochs_remaining == 0
            and self.stabilization_epochs == 0
            and self.cooldown_remaining == 0
        ):
            early_stop_triggered = True
            d.msg_parts.append(
                "[EARLY STOPPING] Patience exceeded "
                f"({epochs_no_improve}/{self.plateau_patience}). Fold exhausted."
            )

        final_msg = f"[LAUNCH] [{phase}] " + " | ".join(d.msg_parts) if d.msg_parts else ""

        return (*d.flags(), early_stop_triggered, final_msg)

    def _tick_metric_focus(self) -> None:
        """Apply the active metric-focus loss weights and count the focus window down."""
        self.metric_focus_epochs_remaining -= 1
        if self.metric_focus_target == "srcc":
            self.current_spearman_weight = 3.0
            self.current_rank_weight = 2.0
        elif self.metric_focus_target == "plcc":
            self.current_emd_weight = 1.5
            self.current_rank_weight = 0.3
            self.current_spearman_weight = 0.2
        elif self.metric_focus_target == "ssim":
            self.current_ssim_weight = 0.15
        elif self.metric_focus_target == "lpips":
            self.current_lpips_weight = 0.08
        elif self.metric_focus_target == "fid":
            self.current_stress = 5.0
        elif self.metric_focus_target == "mae":
            self.current_huber_delta = 0.1
        elif self.metric_focus_target in ("dir_acc", "win_rate"):
            self.current_dir_weight = 2.0
            if self.metric_focus_target == "win_rate":
                self.current_conf_gate_str = 1.5
        elif self.metric_focus_target == "profit_factor":
            self.current_dir_weight = 2.0
            self.current_mag_weight = 0.2
        elif self.metric_focus_target == "max_drawdown":
            self.current_mag_weight = 0.1
            self.current_conf_gate_str = 2.0
        elif self.metric_focus_target in ("tp_mae", "sl_mae"):
            self.current_mag_weight = 1.2

        self.lr_multiplier = self.cooling_factor
        self.head_lr_multiplier = min(self.lr_multiplier * 2.0, 1.5)
        self.jolt_window_remaining = 5

        if self.metric_focus_epochs_remaining == 0:
            self.metric_focus_target = None

    def _force_jump(self) -> AuditResult:
        """Handle a SOTA-forced resolution jump (Monotonic Ratchet via ``_commit_rung``)."""
        epochs_at_res = self.epoch_count - self.last_res_jump_epoch
        if epochs_at_res < self.manifold_maturity:
            return (
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                (
                    f"[GUARD] [HARDENING] SOTA hit early, but locking at {self.current_res}px "
                    "for weight stabilization (Manifold Maturity: "
                    f"{epochs_at_res}/{self.manifold_maturity})."
                ),
            )

        current_idx = (
            self.res_ladder.index(self.current_res) if self.current_res in self.res_ladder else -1
        )
        if 0 <= current_idx < len(self.res_ladder) - 1:
            next_res = self.res_ladder[current_idx + 1]
            self._commit_rung(next_res, 0.5)
            self.last_res_jump_epoch = self.epoch_count
            self.spatial_lock_remaining = self.stabilization_lock
            self.stabilization_epochs = self.stabilization_lock
            self.history.clear()
            return (
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                f"[LAUNCH] [SOTA-FORCE] Jumping to {next_res}px Manifold...",
            )
        return (
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            "[SUCCESS] [SOTA-MAX] Already at maximum resolution.",
        )

    def _track_recovery_streak(self, sentinel_trigger_rate: float) -> None:
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

    def _stabilization_hold(
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
        else:
            self.stabilization_epochs -= 1
            recovery_delta = self.min_delta if self.task_type != "quality" else 0.05
            if current_quality - self.prev_quality > recovery_delta:
                self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                self.stabilization_epochs = max(0, self.stabilization_epochs - 1)
                msg_parts.append(
                    "[RECOVERY] Rapid quality gain detected. Cooldown and Lock shortened."
                )

            self.prev_quality = current_quality
            if current_loss:
                self.prev_loss = current_loss
            status_msg = (
                f"[SIGNAL] Anchoring Manifold... (Cooldown: {self.cooldown_remaining})"
                if self.cooldown_remaining > 0
                else "[SIGNAL] Anchoring Manifold..."
            )
            if msg_parts:
                status_msg = " | ".join(msg_parts) + " | " + status_msg
            return status_msg
        return None

    def _detect_overfitting(self) -> bool:
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

    def _detect_turbulence(self) -> bool:
        """Return True when the last three quality deltas alternate in sign."""
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i - 1] for i in range(1, len(q_values))]
            if all(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas))) and all(
                abs(d) > self.min_delta * 2 for d in deltas
            ):
                return True
        return False

    def _apply_retreat_or_cooling(
        self,
        d: AuditDirectives,
        should_retreat: bool,
        sentinel_trigger_rate: float,
        current_state: tuple[int, float],
        failures: int,
    ) -> None:
        """Record an NPP failure and cool LR, or cool temperature under sentinel stress."""
        if should_retreat:
            self.failure_log[str(current_state)] = failures + 1
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

    def _tick_jolt_window(
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
