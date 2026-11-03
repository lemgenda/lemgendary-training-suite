"""Per-epoch audit of the smart training governor (verdicts and directives)."""

from __future__ import annotations

from dataclasses import dataclass

from training.governance.epoch_signals import (
    NO_CHANGE,
    AuditDirectives,
    AuditResult,
    EpochSignalsMixin,
)
from training.governance.propulsion import PropulsionMixin

__all__ = ["AuditDirectives", "AuditResult", "EpochAuditMixin", "EpochSignals"]


@dataclass(frozen=True)
class EpochSignals:
    """Validation readings of one epoch, as passed to ``audit_epoch``."""

    current_quality: float
    current_loss: float | None = None
    train_loss: float | None = None
    plcc: float = 0.0
    srcc: float = 0.0
    target_std: float | None = None
    metrics_dict: dict[str, float] | None = None


@dataclass
class EpochTrend:
    """Quality trend of one epoch; the two shields may clear the last two flags."""

    delta_q: float
    is_plateaued: bool
    is_flat_leg: bool
    is_flat: bool
    is_trapped: bool
    is_regressing: bool
    is_turbulent: bool


@dataclass(frozen=True)
class EpochVerdict:
    """Phase-level findings of one epoch that do not change during the audit."""

    phase: str
    is_overfitting: bool
    is_collapsed: bool
    loss_is_stable: bool

    @property
    def is_expanding(self) -> bool:
        """True while the curriculum is still growing the data fraction."""
        return self.phase in ["FOUNDATION", "EXPANSION"]


class EpochAuditMixin(EpochSignalsMixin, PropulsionMixin):
    """Epoch audit entry point for ``SmartTrainingGovernor``."""

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
        signals: EpochSignals,
        epochs_no_improve: int,
        sentinel_trigger_rate: float = 0.0,
        force_jump: bool = False,
    ) -> AuditResult:
        """Audit single epoch performance and return optimization directives.

        Returns:
            Tuple of (f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed,
            early_stop_triggered, status_msg)
        """
        if not self.enabled and not force_jump:
            return (*NO_CHANGE, "")

        self.advance_counters()
        if force_jump:
            return self._force_jump()

        self.track_recovery_streak(sentinel_trigger_rate)
        if self.cooldown_remaining > 0:
            self.sota.cooldown_remaining -= 1
        self.sota.record_epoch(signals.current_quality, signals.current_loss, signals.train_loss)

        d = AuditDirectives()
        held = self._held_status(signals, d)
        if held is not None:
            return (*NO_CHANGE, held)

        verdict = self._verdict(signals)
        trend = self._read_trend(signals.current_quality, epochs_no_improve, d)
        self._thermal_shock(signals, d)
        self._spatial_lock(signals.current_loss, trend, d)
        current_state = (self.current_res, round(self.current_fraction, 2))
        should_retreat = self._should_retreat(trend, verdict, d)
        self.apply_retreat_or_cooling(d, should_retreat, sentinel_trigger_rate, current_state)

        trigger_propulsion = self._propulsion_triggered(
            signals.current_quality, trend, verdict, should_retreat
        )
        if self.jolt_window_remaining > 0:
            self.tick_jolt_window(d, trend.delta_q, should_retreat, trend.is_regressing)
        if trigger_propulsion:
            self.propel(d, verdict.phase, signals, trend)
        if (
            not (trend.is_regressing or trend.is_turbulent or sentinel_trigger_rate > 0.15)
            and self.current_temp > self.min_temp
        ):
            self.sharpen(d, verdict.phase, current_state)

        self.remember_epoch(signals.current_quality, signals.current_loss)
        if self.task_type == "quality":
            if self.cooldown_remaining == 0:
                self.thermal.temperature = min(1.0, self.current_temp)
            self.runtime.calibration.current_clamp = min(self.clamp_range[1], self.current_clamp)

        early_stop_triggered = self._early_stop(epochs_no_improve, d)
        final_msg = f"[LAUNCH] [{verdict.phase}] " + " | ".join(d.msg_parts) if d.msg_parts else ""
        return (*d.flags(), early_stop_triggered, final_msg)

    def _held_status(self, signals: EpochSignals, d: AuditDirectives) -> str | None:
        """Return the status of a stabilization hold or resumption shield, else ``None``."""
        if self.stabilization_epochs > 0:
            held = self.stabilization_hold(
                signals.current_quality, signals.current_loss, d.msg_parts
            )
            if held is not None:
                return held

        is_resuming = self.session_epoch_count == 1 and self.epoch_count > 1
        is_regressing_shock = (
            signals.current_quality < self.best_quality * 0.95 and self.best_quality > 0
        )
        if is_resuming and is_regressing_shock:
            self.remember_epoch(signals.current_quality, signals.current_loss)
            return "[GUARD] [SHIELD] Resumption Shield Active. Buffering Momentum Shock."
        return None

    def _verdict(self, signals: EpochSignals) -> EpochVerdict:
        """Detect overfitting and collapse, reset the LR multiplier and read the phase."""
        is_overfitting = self.detect_overfitting()
        self.runtime.lr.lr_multiplier = 1.0
        if self.task_type == "forex":
            is_collapsed = signals.current_quality < 45.0
        else:
            is_collapsed = (signals.current_quality < 0.05) or (signals.plcc < -0.1)
        current_loss = signals.current_loss
        loss_is_stable = (
            (current_loss <= self.prev_loss * 1.05) if current_loss and self.prev_loss else True
        )
        return EpochVerdict(self.get_phase(), is_overfitting, is_collapsed, loss_is_stable)

    def _read_trend(
        self, current_quality: float, epochs_no_improve: int, d: AuditDirectives
    ) -> EpochTrend:
        """Classify the quality movement as plateaued, flat, trapped or regressing."""
        delta_q = current_quality - self.prev_quality
        is_turbulent = self.detect_turbulence()

        is_plateaued = epochs_no_improve >= self.plateau_patience
        is_flat_leg = abs(delta_q) < self.min_delta and len(self.history) >= 2

        abs_floor = 40.0 if self.task_type == "quality" else 0.0
        fidelity_floor = max(abs_floor, self.best_quality * 0.8)
        is_trapped = current_quality < fidelity_floor and len(self.history) >= 4
        if is_trapped:
            is_flat_leg = abs(delta_q) < self.min_delta * 4
            if is_plateaued or is_flat_leg:
                d.msg_parts.append("[TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        regress_threshold = (
            -0.03 if self.task_type in ["quality", "parameter_prediction"] else -0.01
        )
        is_regressing = (
            delta_q < (self.prev_quality * regress_threshold) if self.prev_quality else False
        )
        return EpochTrend(
            delta_q=delta_q,
            is_plateaued=is_plateaued,
            is_flat_leg=is_flat_leg,
            is_flat=is_plateaued or is_flat_leg,
            is_trapped=is_trapped,
            is_regressing=is_regressing,
            is_turbulent=is_turbulent,
        )

    def _thermal_shock(self, signals: EpochSignals, d: AuditDirectives) -> None:
        """Sharpen the temperature when PLCC turns negative on a non-degenerate target."""
        if (
            signals.plcc < -0.01
            and self.current_temp > 0.7
            and (signals.target_std is None or signals.target_std >= 0.15)
        ):
            self.thermal.temperature = 0.5
            self.runtime.calibration.current_clamp = 20.0
            d.t_changed = d.c_changed = True
            d.msg_parts.append("[THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

    def _spatial_lock(
        self, current_loss: float | None, trend: EpochTrend, d: AuditDirectives
    ) -> None:
        """Count the spatial lock down; it buffers a regression unless the loss explodes."""
        if self.spatial_lock_remaining <= 0:
            return
        self.runtime.counters.spatial_lock_remaining -= 1
        loss_is_exploding = (
            (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
        )
        if trend.is_regressing and not loss_is_exploding:
            trend.is_regressing = False
            d.msg_parts.append(
                f"[SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})"
            )

    def _should_retreat(self, trend: EpochTrend, verdict: EpochVerdict, d: AuditDirectives) -> bool:
        """Decide whether the epoch counts as an NPP failure, applying the shields."""
        if self.task_type in ["quality", "forex"]:
            trend.is_turbulent = False
            if verdict.is_expanding:
                d.msg_parts.append(
                    "[SIGNAL] [RESONANCE] Turbulence detected but shielded. Holding manifold."
                )

        should_retreat = trend.is_regressing or trend.is_turbulent or verdict.is_collapsed
        if verdict.is_expanding and verdict.loss_is_stable and not verdict.is_collapsed:
            should_retreat = False
            if trend.is_regressing:
                d.msg_parts.append(
                    "[GUARD] [MOMENTUM] Jitter detected but Loss is stable. Holding manifold."
                )

        if verdict.is_overfitting and (verdict.is_expanding or self.current_fraction >= 1.0):
            self.sota.cooldown_remaining = 0
            self._overfitting_rescue(verdict.is_expanding, d)
            return False
        return should_retreat

    def _overfitting_rescue(self, is_expanding: bool, d: AuditDirectives) -> None:
        """Expand the data while growing, else raise the stress protocol level."""
        if is_expanding:
            d.msg_parts.append(
                "[RESCUE] [OVERFITTING] Overfitting detected. Forcing dataset expansion to "
                "introduce variety."
            )
        elif self.task_type == "forex":
            self.sota.current_stress = min(2.0, self.current_stress + 0.5)
            d.msg_parts.append(
                "[REGULARIZE] [OVERFITTING] Temporal fold divergence detected. Applying mild "
                f"stabilization (Level {self.current_stress})."
            )
        else:
            self.sota.current_stress = min(5.0, self.current_stress + 1.0)
            d.msg_parts.append(
                "[RESCUE] [OVERFITTING] Dataset exhausted. Deploying Stress Protocol (Level "
                f"{self.current_stress})."
            )

    def _propulsion_triggered(
        self,
        current_quality: float,
        trend: EpochTrend,
        verdict: EpochVerdict,
        should_retreat: bool,
    ) -> bool:
        """Return True when the curriculum may advance this epoch."""
        if should_retreat or self.cooldown_remaining != 0:
            return False
        stride_threshold = (
            0.75 if (self.current_res is not None and self.current_res < 512) else 0.90
        )
        if self.target_quality_score > 1.0:
            stride_threshold = stride_threshold * self.target_quality_score

        epochs_since_step = self.epoch_count - max(self.last_res_jump_epoch, self.last_action_epoch)
        min_soak = max(3, self.manifold_maturity)
        fraction_soak_complete = verdict.is_expanding and epochs_since_step >= min_soak
        deepening_ready = (
            verdict.phase == "DEEPENING"
            and current_quality >= stride_threshold
            and epochs_since_step >= min_soak
        )
        if verdict.is_overfitting and verdict.is_expanding:
            return True
        if trend.is_plateaued and (fraction_soak_complete or not verdict.is_expanding):
            return True
        return trend.delta_q >= -self.min_delta and (
            (trend.is_flat_leg and not verdict.is_expanding)
            or fraction_soak_complete
            or deepening_ready
        )

    def _early_stop(self, epochs_no_improve: int, d: AuditDirectives) -> bool:
        """Return True when patience is exhausted outside every focus, lock and cooldown."""
        if (
            epochs_no_improve >= self.plateau_patience
            and self.metric_focus_epochs_remaining == 0
            and self.stabilization_epochs == 0
            and self.cooldown_remaining == 0
        ):
            d.msg_parts.append(
                "[EARLY STOPPING] Patience exceeded "
                f"({epochs_no_improve}/{self.plateau_patience}). Fold exhausted."
            )
            return True
        return False

    def _force_jump(self) -> AuditResult:
        """Handle a SOTA-forced resolution jump (Monotonic Ratchet via ``_commit_rung``)."""
        epochs_at_res = self.epoch_count - self.last_res_jump_epoch
        if epochs_at_res < self.manifold_maturity:
            hold_msg = (
                f"[GUARD] [HARDENING] SOTA hit early, but locking at {self.current_res}px "
                "for weight stabilization (Manifold Maturity: "
                f"{epochs_at_res}/{self.manifold_maturity})."
            )
            return (*NO_CHANGE, hold_msg)

        current_idx = (
            self.res_ladder.index(self.current_res) if self.current_res in self.res_ladder else -1
        )
        if 0 <= current_idx < len(self.res_ladder) - 1:
            next_res = self.res_ladder[current_idx + 1]
            self._commit_rung(next_res, 0.5)
            self.runtime.counters.last_res_jump_epoch = self.epoch_count
            self.runtime.counters.spatial_lock_remaining = self.stabilization_lock
            self.sota.stabilization_epochs = self.stabilization_lock
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
        return (*NO_CHANGE, "[SUCCESS] [SOTA-MAX] Already at maximum resolution.")
