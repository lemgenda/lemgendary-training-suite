"""Curriculum propulsion steps taken by the governor once an audit allows progress."""

from __future__ import annotations

from training.governance.epoch_audit import AuditDirectives
from training.ratchet import (
    expand_fraction,
)


class PropulsionMixin:
    """Curriculum propulsion steps taken by the governor once an audit allows progress."""

    def _identify_lagging_metric(self, metrics_dict: dict[str, float]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for key, target in self.sota_targets.items():
            current = metrics_dict.get(key)
            if current is None or target is None or target == 0:
                continue
            is_higher_better = self.metrics.is_higher_better(key)
            if is_higher_better:
                deficit = max(0.0, (target - current) / abs(target))
            else:
                deficit = max(0.0, (current - target) / abs(target))
            if deficit > 0:
                results.append((key, deficit))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _apply_targeted_optimizations(
        self,
        lagging_list: list[tuple[str, float]],
    ) -> list[str]:
        msg_parts: list[str] = []
        if not lagging_list:
            return msg_parts

        primary_metric, deficit = lagging_list[0]

        # Apply targeted stabilization based on deficit
        if primary_metric in ["srcc", "plcc"]:
            self.current_spearman_weight = min(3.0, self.current_spearman_weight * 1.5)
            self.current_rank_weight = min(2.5, self.current_rank_weight * 1.3)
            msg_parts.append(
                f"TARGETED: Boosting rank weights for {primary_metric} (deficit={deficit:.2%})"
            )
        elif primary_metric in ["dir_acc", "win_rate"]:
            self.current_dir_weight = min(3.0, self.current_dir_weight * 1.4)
            msg_parts.append(f"TARGETED: Elevating directional weight for {primary_metric}")

        return msg_parts

    def _plateau_jolt(
        self, d: AuditDirectives, phase: str, is_flat: bool, is_trapped: bool
    ) -> None:
        """Fire a head-differential LR jolt when the plateau cooldown allows it."""
        jolt_ready = (
            self.epoch_count - self.last_jolt_epoch
        ) > self.jolt_cooldown and self.cooldown_remaining == 0
        if is_flat and jolt_ready and self.jolt_window_remaining == 0:
            jolt = self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5)
            if is_trapped:
                jolt *= 1.5
            if self.task_type == "forex":
                jolt *= 2.0
            jolt_base = float(jolt)

            if phase == "REFINEMENT":
                self.lr_multiplier = round(jolt_base * 0.5, 3)
                self.head_lr_multiplier = min(
                    round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3)
                )
            else:
                self.lr_multiplier = jolt_base
                self.head_lr_multiplier = jolt_base

            window_size = 5 if self.task_type == "forex" else 3
            self.jolt_window_remaining = window_size
            d.lr_changed = True
            self.last_jolt_epoch = self.epoch_count
            d.msg_parts.append(
                "JOLT: Breaking Plateau with Head-Differential Propulsion (Head: "
                f"{self.head_lr_multiplier:.2f}x, Backbone: {self.lr_multiplier:.2f}x | "
                f"{window_size}-Epoch Window)"
            )

    def _expand_fraction(
        self, d: AuditDirectives, next_frac: float, current_quality: float
    ) -> None:
        """FOUNDATION/EXPANSION propulsion: grow the data fraction at the same resolution."""
        if self.current_fraction < 1.0:
            self._commit_rung(self.current_res, next_frac)
            self.last_action_epoch = self.epoch_count
            d.f_changed = True
            d.msg_parts.append(f"PROPULSION: Data -> {self.current_fraction * 100:.0f}%")
            cooldown_soak = max(3, self.manifold_maturity)
            self.stabilization_epochs = cooldown_soak
            self.cooldown_remaining = cooldown_soak
            self.best_quality = current_quality

    def _spatial_jump(self, d: AuditDirectives) -> None:
        """DEEPENING propulsion: step up one resolution rung and reset the data fraction."""
        current_idx = self.res_ladder.index(self.current_res)
        next_res = self.res_ladder[current_idx + 1]

        if self.best_quality < self.target_quality_score * 0.80 and self.target_quality_score > 1.0:
            self.lr_multiplier = self.cooling_factor
            self.head_lr_multiplier = self.cooling_factor
            d.lr_changed = True
            d.msg_parts.append("RECOIL: Insufficient Quality for spatial jump. Cooling LR.")
            self.stabilization_epochs = self.stabilization_lock
        else:
            init_frac = float(
                self.model_info.get("optimization", {}).get(
                    "initial_fraction",
                    self.config.get("governor", {})
                    .get("manifold", {})
                    .get("initial_fraction", 0.15),
                )
            )
            self._commit_rung(next_res, init_frac)
            d.r_changed = d.b_changed = True
            d.f_changed = True
            self.last_res_jump_epoch = self.epoch_count
            self.last_action_epoch = self.epoch_count
            self.spatial_lock_remaining = self.stabilization_lock
            d.msg_parts.append(
                f"SPATIAL JUMP: {next_res}px | Data Reset {init_frac * 100:.0f}% | Lock: ON"
            )
            self.stabilization_epochs = self.stabilization_lock

    def _refine(
        self,
        d: AuditDirectives,
        metrics_dict: dict[str, float] | None,
        plcc: float,
        srcc: float,
    ) -> None:
        """REFINEMENT propulsion: stress protocol, precision cooling or targeted metric focus."""
        if self.task_type == "forex":
            if (
                self.current_stress < 2.0
                and self.target_quality_score > 0
                and self.best_quality < self.target_quality_score * 0.90
            ):
                self.current_stress = min(2.0, self.current_stress + 0.5)
                self.lr_multiplier = 1.0
                self.head_lr_multiplier = 1.10
                self.jolt_window_remaining = 2
                d.lr_changed = True
                d.msg_parts.append(
                    "REFINEMENT: Plateau in Temporal Fold. Deploying Gentle Fold Tuning "
                    "(Head LR: 1.10x)"
                )
            else:
                if self.jolt_window_remaining == 0:
                    self.lr_multiplier = self.cooling_factor
                    self.head_lr_multiplier = self.cooling_factor
                    d.lr_changed = True
                    d.msg_parts.append("REFINEMENT: Precision Cooling")
        elif (
            self.current_stress < 5.0
            and self.target_quality_score > 0
            and self.best_quality < self.target_quality_score * 0.90
        ):
            self.current_stress = min(5.0, self.current_stress + 1.0)
            jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
            self.lr_multiplier = round(jolt_base * 0.5, 3)
            self.head_lr_multiplier = min(
                round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3)
            )
            self.jolt_window_remaining = 3
            d.lr_changed = True
            self.max_stress_stuck_epochs = 0
            d.msg_parts.append(
                "REFINEMENT: Trapped in Plateau. Deploying Stress Protocol (Level "
                f"{self.current_stress}) & Differential Jolt"
            )
        elif (
            self.current_stress >= 5.0
            and self.target_quality_score > 0
            and self.best_quality < self.target_quality_score * 0.90
        ):
            jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
            self.lr_multiplier = jolt_base
            self.head_lr_multiplier = jolt_base
            d.lr_changed = True
            self.max_stress_stuck_epochs = self.max_stress_stuck_epochs + 1
            stuck_patience = self.plateau_patience * 2
            d.msg_parts.append(
                f"REFINEMENT: [MAX STRESS] Forcing Jolt (x{jolt_base:.2f}) to maintain "
                f"momentum (Stuck: {self.max_stress_stuck_epochs}/{stuck_patience})"
            )
            if self.max_stress_stuck_epochs >= stuck_patience:
                self.trigger_mini_swa = True
                d.msg_parts.append(
                    "[MINI-SWA PULSE] Triggering weight averaging pulse across top "
                    f"checkpoints (Stuck: {self.max_stress_stuck_epochs} epochs)"
                )
        else:
            if self.jolt_window_remaining == 0:
                self.lr_multiplier = self.cooling_factor
                self.head_lr_multiplier = self.cooling_factor
                d.lr_changed = True
                d.msg_parts.append("REFINEMENT: SOTA Precision Cooling")

            if metrics_dict is None:
                metrics_dict = {"plcc": plcc, "srcc": srcc}

            lagging = self._identify_lagging_metric(metrics_dict)
            if lagging:
                adjustments = self._apply_targeted_optimizations(lagging)
                d.msg_parts.extend(adjustments)

    def propel(
        self,
        d: AuditDirectives,
        phase: str,
        current_quality: float,
        is_flat: bool,
        is_trapped: bool,
        metrics_dict: dict[str, float] | None,
        plcc: float,
        srcc: float,
    ) -> None:
        """Advance the curriculum one step for the current phase."""
        self._plateau_jolt(d, phase, is_flat, is_trapped)

        next_frac = expand_fraction(self.current_fraction, self.fraction_increment)
        next_state = (self.current_res, round(next_frac, 2))

        if self.failure_log.get(str(next_state), 0) > 0:
            self.lr_multiplier = 0.6
            self.head_lr_multiplier = 0.6
            d.lr_changed = True
            d.msg_parts.append("ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")

        if phase in ("FOUNDATION", "EXPANSION"):
            self._expand_fraction(d, next_frac, current_quality)
        elif (
            phase == "DEEPENING"
            and self.current_res is not None
            and self.current_res in self.res_ladder
        ):
            self._spatial_jump(d)
        else:
            self._refine(d, metrics_dict, plcc, srcc)

    def sharpen(self, d: AuditDirectives, phase: str, current_state: tuple[int, float]) -> None:
        """Lower the softmax temperature toward the phase floor when cooldown allows."""
        if self.task_type == "forex":
            phase_min = self.min_temp
        elif self.task_type != "quality":
            phase_min = 0.05 if phase == "REFINEMENT" else 0.1
        else:
            phase_min = self.min_temp

        floor = max(phase_min, self.thermal_floor.get(str(current_state), self.min_temp))

        if self.cooldown_remaining == 0 and self.current_temp > floor:
            sharpen_rate = 0.95 if self.current_temp > 1.2 else self.sharpening_rate
            self.current_temp = max(floor, self.current_temp * sharpen_rate)
            d.t_changed = True
            d.msg_parts.append(f"SHARPENING: Temp -> {self.current_temp:.2f}")
        elif self.cooldown_remaining > 0:
            d.msg_parts.append(f"MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")
