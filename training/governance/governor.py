"""Central governor coordinating curriculum, thermal, metrics, and SOTA tracking."""

from __future__ import annotations

import logging
from typing import Any

import torch

from training.governance.curriculum import CurriculumState
from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS, DEFAULT_METRIC_WEIGHTS, MetricRegistry
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState

logger = logging.getLogger("lemtrain.governance.governor")


class GovernorStateError(Exception):
    """Raised when an invalid state transition or operation occurs in the governor."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class SmartTrainingGovernor:
    """Universal Autonomous Optimization Engine (v15.6 Nuclear).

    Coordinates curriculum resolution ladders, thermal decay, metric monitoring,
    and SOTA stability management.
    """

    def __init__(
        self,
        model_info: dict[str, Any],
        config: dict[str, Any] | None = None,
        stabilizers: dict[str, Any] | None = None,
    ) -> None:
        self.model_info = model_info
        self.config = config or {}
        opt = model_info.get("optimization", {})
        manifold_defaults = self.config.get("governor", {}).get("manifold", {})

        self.enabled: bool = opt.get("enabled", True)

        # Initialize sub-states
        self.metrics = MetricRegistry()
        self.curriculum = CurriculumState()
        self.thermal = ThermalState()
        self.sota = SotaTracker()

        # Curriculum configuration
        res_ladder = opt.get("res_ladder")
        self.curriculum.target_effective_batch = opt.get(
            "target_effective_batch",
            manifold_defaults.get("target_effective_batch", 24),
        )
        self.manifold_maturity: int = opt.get(
            "manifold_maturity",
            manifold_defaults.get("maturity_soak", 5),
        )
        self.sota.plateau_patience = opt.get(
            "plateau_patience",
            self.config.get("governor", {}).get("plateau_patience", 6),
        )

        self.plateau_priority: str = opt.get("plateau_priority", "data")
        self.curriculum.fraction_increment = opt.get(
            "fraction_increment",
            manifold_defaults.get("fraction_increment", 0.15),
        )
        self.cooling_factor: float = opt.get("cooling_factor", 0.5)
        self.clamp_range: list[float] = opt.get("clamp_range", [15.0, 45.0])
        self.thermal.clamp_range = list(self.clamp_range)

        gov_cfg = self.config.get("governor", {})
        self.jolt_cooldown: int = gov_cfg.get("jolt_cooldown_epochs", 5)
        self.stabilization_lock: int = gov_cfg.get("stabilization_lock_epochs", 3)
        self.breakout_threshold: float = gov_cfg.get("emergency_breakout_threshold", 0.10)
        self.sharpening_rate: float = gov_cfg.get("sharpening_cooling_rate", 0.98)

        # Fraction and batch setup
        initial_fraction = model_info.get("sample_fraction") or opt.get(
            "initial_fraction",
            manifold_defaults.get("initial_fraction", 0.15),
        )
        self.curriculum.current_fraction = float(initial_fraction)

        batch_val = model_info.get("batch_size", 16)
        self.curriculum.current_batch = int(batch_val) if batch_val and batch_val != "auto" else 16
        self.curriculum.current_acc = 1

        raw_size = model_info.get("input_size", 224)
        current_res = raw_size[1] if isinstance(raw_size, (list, tuple)) else raw_size
        self.curriculum.current_res = int(current_res)

        if not res_ladder:
            stride = manifold_defaults.get("resolution_stride", 128)
            max_res = manifold_defaults.get("max_resolution", 1024)
            constructed_ladder: list[int] = []
            curr = self.curriculum.current_res
            while curr <= max_res:
                constructed_ladder.append(curr)
                curr += stride
            if not constructed_ladder:
                constructed_ladder = [self.curriculum.current_res]
            self.curriculum.res_ladder = constructed_ladder
        else:
            self.curriculum.res_ladder = list(res_ladder)

        self.stab = stabilizers or {}
        raw_task = model_info.get("dataset_type", "quality")
        self.task_type: str = raw_task[0] if isinstance(raw_task, list) else str(raw_task)
        self.sota.task_type = self.task_type

        if self.task_type == "forex":
            self.sota.plateau_patience = max(self.sota.plateau_patience, 15)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8.0
        if self.task_type in ["restoration", "enhancement", "face", "parameter_prediction"]:
            if vram_gb < 4.5:
                max_safe_res = 256 if self.task_type == "parameter_prediction" else 384
            elif vram_gb < 8.5:
                max_safe_res = 384
            elif vram_gb < 16.5:
                max_safe_res = 512
            else:
                max_safe_res = 1024

            cfg_max_res = self.config.get("hardware", {}).get("max_allowed_resolution")
            if cfg_max_res:
                max_safe_res = min(max_safe_res, int(cfg_max_res))

            self.curriculum.res_ladder = [r for r in self.curriculum.res_ladder if r <= max_safe_res]
            if not self.curriculum.res_ladder:
                self.curriculum.res_ladder = [max_safe_res]
            if self.curriculum.current_res is not None and self.curriculum.current_res > max_safe_res:
                logger.info(
                    "Hardware VRAM Cap Active (%.1fGB): Clamping %dpx -> %dpx for memory stability.",
                    vram_gb,
                    self.curriculum.current_res,
                    max_safe_res,
                )
                self.curriculum.current_res = max_safe_res

        if self.curriculum.res_ladder and self.curriculum.current_res != self.curriculum.res_ladder[0] and self.task_type != "forex":
            logger.info(
                "Aligning start resolution %dpx -> lowest rung %dpx.",
                self.curriculum.current_res,
                self.curriculum.res_ladder[0],
            )
            self.curriculum.current_res = self.curriculum.res_ladder[0]

        if self.curriculum.current_res is not None and self.curriculum.current_res not in self.curriculum.res_ladder:
            self.curriculum.res_ladder = sorted(list(set(self.curriculum.res_ladder + [self.curriculum.current_res])))

        if self.task_type == "forex":
            self.min_temp = max(0.75, float(self.stab.get("min_temp", 0.75)))
            self.current_temp = max(self.min_temp, float(self.stab.get("softmax_temp", 1.0)))
        else:
            default_min = 0.5 if self.task_type == "quality" else (0.01 if self.task_type == "parameter_prediction" else 0.1)
            self.min_temp = float(self.stab.get("min_temp", default_min))
            self.current_temp = float(self.stab.get("softmax_temp", self.min_temp))
        self.thermal.temperature = self.current_temp
        self.thermal.min_temperature = self.min_temp
        self.current_clamp = float(self.stab.get("logit_clamp", 15.0))

        self.current_rank_weight = float(self.stab.get("rank_weight", 0.8))
        self.current_rank_margin = float(self.stab.get("rank_margin", 0.10))
        self.max_rank_weight = float(self.stab.get("max_rank_weight", 1.5))
        self.min_rank_margin = float(self.stab.get("min_rank_margin", 0.05))
        self.current_spearman_weight = float(self.stab.get("soft_spearman_weight", 0.5))

        self.current_emd_weight = float(self.stab.get("emd_weight", 1.0))
        self.current_ssim_weight = float(self.stab.get("ssim_weight", 0.0))
        self.current_huber_delta = float(self.stab.get("huber_delta", 1.0))
        self.current_conf_gate_str = float(self.stab.get("conf_gate_strength", 1.0))
        self.current_lpips_weight = float(self.stab.get("lpips_weight", 0.025))
        self.current_mag_weight = float(self.stab.get("mag_weight", 0.5))
        self.current_dir_weight = float(self.stab.get("dir_weight", 1.0))

        self.metric_focus_epochs_remaining: int = 0
        self.metric_focus_target: str | None = None
        self.metric_focus_last_fired: dict[str, int] = {}

        self.sota.loop_breaker_enabled = opt.get("loop_breaker_enabled", True)
        self.sota.loop_breaker_threshold = opt.get("loop_breaker_threshold", 2)
        self.sota.loop_breaker_strategy = opt.get("loop_breaker_strategy", "auto")

        self.thermal_floor: dict[str, float] = {}
        self.lr_multiplier: float = 1.0
        self.head_lr_multiplier: float = 1.0
        self.jolt_window_remaining: int = 0
        self.trigger_mini_swa: bool = False
        self.last_action_epoch: int = 0
        self.epoch_count: int = 0
        self.session_epoch_count: int = 0
        self.last_jolt_epoch: int = -10

        base_delta = opt.get("min_delta", 0.0005)
        if self.task_type == "parameter_prediction":
            self.min_delta = base_delta
        else:
            self.min_delta = base_delta if self.task_type == "quality" else (base_delta * 100.0)
        self.spatial_lock_remaining: int = 0
        self.last_res_jump_epoch: int = 0

        self.sota_targets: dict[str, float] = opt.get("sota_targets", model_info.get("sota_targets", {}))
        self.sota.sota_targets = self.sota_targets
        self.sota.target_quality_score = 1.0
        if self.sota_targets:
            target_score = 0.0
            for k, target_v in self.sota_targets.items():
                direction = DEFAULT_METRIC_DIRECTIONS.get(k, True)
                weight = DEFAULT_METRIC_WEIGHTS.get(k, 1.0)
                if direction:
                    target_score += target_v * weight
                else:
                    if k == "fid":
                        target_score += (100.0 - target_v) * weight
                    elif k == "lpips":
                        target_score += (1.0 - target_v) * weight
                    elif k == "rank_margin":
                        target_score += (10.0 - target_v) * weight
                    elif k == "max_drawdown":
                        target_score += max(0.0, 100.0 - target_v) * weight
                    elif k in ["tp_mae", "sl_mae"]:
                        target_score += max(0.0, 50.0 - target_v) * weight
                    elif k == "dir_entropy":
                        target_score += max(0.0, 1.099 - target_v) * weight
                    else:
                        target_score += (1.0 / (target_v + 1e-6)) * weight
            if target_score > 0:
                if target_score <= 1.0 and self.task_type == "quality":
                    target_score *= 100.0
                self.sota.target_quality_score = target_score

    # -------------------------------------------------------------------------
    # Delegated Properties
    # -------------------------------------------------------------------------
    @property
    def current_fraction(self) -> float:
        return self.curriculum.current_fraction

    @current_fraction.setter
    def current_fraction(self, value: float) -> None:
        self.curriculum.current_fraction = float(value)

    @property
    def current_res(self) -> int:
        return self.curriculum.current_res

    @current_res.setter
    def current_res(self, value: int) -> None:
        self.curriculum.current_res = int(value)

    @property
    def res_ladder(self) -> list[int]:
        return self.curriculum.res_ladder

    @res_ladder.setter
    def res_ladder(self, value: list[int]) -> None:
        self.curriculum.res_ladder = list(value)

    @property
    def current_batch(self) -> int:
        return self.curriculum.current_batch

    @current_batch.setter
    def current_batch(self, value: int) -> None:
        self.curriculum.current_batch = int(value)

    @property
    def current_acc(self) -> int:
        return self.curriculum.current_acc

    @current_acc.setter
    def current_acc(self, value: int) -> None:
        self.curriculum.current_acc = int(value)

    @property
    def target_effective_batch(self) -> int:
        return self.curriculum.target_effective_batch

    @target_effective_batch.setter
    def target_effective_batch(self, value: int) -> None:
        self.curriculum.target_effective_batch = int(value)

    @property
    def fraction_increment(self) -> float:
        return self.curriculum.fraction_increment

    @fraction_increment.setter
    def fraction_increment(self, value: float) -> None:
        self.curriculum.fraction_increment = float(value)

    @property
    def current_temp(self) -> float:
        return self.thermal.temperature

    @current_temp.setter
    def current_temp(self, value: float) -> None:
        self.thermal.temperature = float(value)

    @property
    def best_quality(self) -> float:
        return self.sota.best_quality

    @best_quality.setter
    def best_quality(self, value: float) -> None:
        self.sota.best_quality = float(value)

    @property
    def prev_quality(self) -> float:
        return self.sota.prev_quality

    @prev_quality.setter
    def prev_quality(self, value: float) -> None:
        self.sota.prev_quality = float(value)

    @property
    def prev_loss(self) -> float:
        return self.sota.prev_loss

    @prev_loss.setter
    def prev_loss(self, value: float) -> None:
        self.sota.prev_loss = float(value)

    @property
    def target_quality_score(self) -> float:
        return self.sota.target_quality_score

    @target_quality_score.setter
    def target_quality_score(self, value: float) -> None:
        self.sota.target_quality_score = float(value)

    @property
    def recovery_streak(self) -> int:
        return self.sota.recovery_streak

    @recovery_streak.setter
    def recovery_streak(self, value: int) -> None:
        self.sota.recovery_streak = int(value)

    @property
    def consecutive_rollbacks(self) -> int:
        return self.sota.consecutive_rollbacks

    @consecutive_rollbacks.setter
    def consecutive_rollbacks(self, value: int) -> None:
        self.sota.consecutive_rollbacks = int(value)

    @property
    def rollback_history(self) -> dict[int, int]:
        return self.sota.rollback_history

    @rollback_history.setter
    def rollback_history(self, value: dict[int, int]) -> None:
        self.sota.rollback_history = dict(value)

    @property
    def gate_relaxation_epochs(self) -> int:
        return self.sota.gate_relaxation_epochs

    @gate_relaxation_epochs.setter
    def gate_relaxation_epochs(self, value: int) -> None:
        self.sota.gate_relaxation_epochs = int(value)

    @property
    def sota_resolution(self) -> int | None:
        return self.sota.sota_resolution

    @sota_resolution.setter
    def sota_resolution(self, value: int | None) -> None:
        self.sota.sota_resolution = value

    @property
    def breakout_lock(self) -> int:
        return self.sota.breakout_lock

    @breakout_lock.setter
    def breakout_lock(self, value: int) -> None:
        self.sota.breakout_lock = int(value)

    @property
    def history(self) -> list[tuple[float, float | None, float | None]]:
        return self.sota.history

    @history.setter
    def history(self, value: list[tuple[float, float | None, float | None]]) -> None:
        self.sota.history = list(value)

    @property
    def failure_log(self) -> dict[str, int]:
        return self.sota.failure_log

    @failure_log.setter
    def failure_log(self, value: dict[str, int]) -> None:
        self.sota.failure_log = dict(value)

    @property
    def stabilization_epochs(self) -> int:
        return self.sota.stabilization_epochs

    @stabilization_epochs.setter
    def stabilization_epochs(self, value: int) -> None:
        self.sota.stabilization_epochs = int(value)

    @property
    def cooldown_remaining(self) -> int:
        return self.sota.cooldown_remaining

    @cooldown_remaining.setter
    def cooldown_remaining(self, value: int) -> None:
        self.sota.cooldown_remaining = int(value)

    @property
    def current_stress(self) -> float:
        return self.sota.current_stress

    @current_stress.setter
    def current_stress(self, value: float) -> None:
        self.sota.current_stress = float(value)

    @property
    def max_stress_stuck_epochs(self) -> int:
        return self.sota.max_stress_stuck_epochs

    @max_stress_stuck_epochs.setter
    def max_stress_stuck_epochs(self, value: int) -> None:
        self.sota.max_stress_stuck_epochs = int(value)

    @property
    def plateau_patience(self) -> int:
        return self.sota.plateau_patience

    @plateau_patience.setter
    def plateau_patience(self, value: int) -> None:
        self.sota.plateau_patience = int(value)

    @property
    def loop_breaker_enabled(self) -> bool:
        return self.sota.loop_breaker_enabled

    @loop_breaker_enabled.setter
    def loop_breaker_enabled(self, value: bool) -> None:
        self.sota.loop_breaker_enabled = bool(value)

    @property
    def loop_breaker_threshold(self) -> int:
        return self.sota.loop_breaker_threshold

    @loop_breaker_threshold.setter
    def loop_breaker_threshold(self, value: int) -> None:
        self.sota.loop_breaker_threshold = int(value)

    @property
    def loop_breaker_strategy(self) -> str:
        return self.sota.loop_breaker_strategy

    @loop_breaker_strategy.setter
    def loop_breaker_strategy(self, value: str) -> None:
        self.sota.loop_breaker_strategy = str(value)

    # -------------------------------------------------------------------------
    # Core Autonomous Operations
    # -------------------------------------------------------------------------
    def suggest_batch_growth(
        self,
        current_batch: int,
        current_acc: int,
        target_eff: int,
        vram_free_ratio: float,
    ) -> tuple[int, int]:
        """Propose doubling physical batch if VRAM headroom > 40%."""
        if current_batch < 1 or vram_free_ratio < 0.40:
            return current_batch, current_acc
        new_batch = min(current_batch * 2, target_eff)
        if new_batch == current_batch:
            return current_batch, current_acc
        new_acc = max(1, target_eff // new_batch)
        return new_batch, new_acc

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
        metrics_dict: dict[str, float],
    ) -> list[str]:
        msg_parts: list[str] = []
        if not lagging_list:
            return msg_parts

        primary_metric, deficit = lagging_list[0]
        severity = "MILD"
        if primary_metric in ["srcc", "plcc", "accuracy", "dir_acc", "win_rate"]:
            severity = "CRITICAL" if deficit > 0.15 else "MILD"
        else:
            severity = "CRITICAL" if deficit > 0.30 else "MILD"

        # Apply targeted stabilization based on deficit
        if primary_metric in ["srcc", "plcc"]:
            self.current_spearman_weight = min(3.0, self.current_spearman_weight * 1.5)
            self.current_rank_weight = min(2.5, self.current_rank_weight * 1.3)
            msg_parts.append(f"TARGETED: Boosting rank weights for {primary_metric} (deficit={deficit:.2%})")
        elif primary_metric in ["dir_acc", "win_rate"]:
            self.current_dir_weight = min(3.0, self.current_dir_weight * 1.4)
            msg_parts.append(f"TARGETED: Elevating directional weight for {primary_metric}")

        return msg_parts

    def get_phase(self) -> str:
        """Determine current training curriculum phase."""
        if self.curriculum.current_fraction < 1.0:
            return "FOUNDATION" if self.curriculum.current_fraction <= 0.40 else "EXPANSION"

        if self.curriculum.res_ladder and self.curriculum.current_res < self.curriculum.res_ladder[-1]:
            return "DEEPENING"

        return "REFINEMENT"

    def audit_epoch(
        self,
        current_quality: float,
        best_quality: float,
        epochs_no_improve: int,
        regression_epochs: int,
        sentinel_trigger_rate: float = 0.0,
        current_lr: float | None = None,
        base_lr: float | None = None,
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
            Tuple of (f_changed, r_changed, lr_changed, t_changed, c_changed, b_changed, early_stop_triggered, status_msg)
        """
        if not self.enabled and not force_jump:
            return False, False, False, False, False, False, False, ""

        self.epoch_count += 1
        self.session_epoch_count += 1

        if self.metric_focus_epochs_remaining > 0:
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

        if self.breakout_lock > 0:
            self.breakout_lock -= 1

        if force_jump:
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
                    f"[GUARD] [HARDENING] SOTA hit early, but locking at {self.current_res}px for weight stabilization (Manifold Maturity: {epochs_at_res}/{self.manifold_maturity}).",
                )

            current_idx = self.res_ladder.index(self.current_res) if self.current_res in self.res_ladder else -1
            if 0 <= current_idx < len(self.res_ladder) - 1:
                next_res = self.res_ladder[current_idx + 1]
                self.current_res = next_res
                self.current_fraction = 0.5
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

        if sentinel_trigger_rate == 0:
            self.recovery_streak += 1
            if self.recovery_streak >= 2 and self.stabilization_epochs > 0 and self.spatial_lock_remaining == 0:
                self.stabilization_epochs = 0
                logger.info("[LAUNCH] [NPP] Stress at zero. Breaking stabilization lock.")
        else:
            self.recovery_streak = 0

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        self.sota.record_epoch(current_quality, current_loss, train_loss)

        msg_parts: list[str] = []
        if self.stabilization_epochs > 0:
            if current_quality < self.best_quality * (1.0 - self.breakout_threshold) and self.best_quality > 0:
                msg_parts.append(f"[BREAKOUT] Shield shattered! Quality dropped {(1-current_quality/self.best_quality)*100:.1f}%.")
                self.stabilization_epochs = 0
            else:
                self.stabilization_epochs -= 1
                recovery_delta = self.min_delta if self.task_type != "quality" else 0.05
                if current_quality - self.prev_quality > recovery_delta:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 2)
                    self.stabilization_epochs = max(0, self.stabilization_epochs - 1)
                    msg_parts.append("[RECOVERY] Rapid quality gain detected. Cooldown and Lock shortened.")

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
                return False, False, False, False, False, False, False, status_msg

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

        is_overfitting = False
        if len(self.history) >= 3:
            train_losses = [h[2] for h in self.history if len(h) > 2 and h[2] is not None]
            val_losses = [h[1] for h in self.history if h[1] is not None]
            if len(train_losses) >= 3 and len(val_losses) >= 3:
                train_trend = train_losses[-1] - train_losses[-3]
                val_trend = val_losses[-1] - val_losses[-3]
                if train_trend < -1e-4 and val_trend > 1e-4:
                    is_overfitting = True

        f_changed = r_changed = lr_changed = t_changed = c_changed = b_changed = False
        self.lr_multiplier = 1.0
        phase = self.get_phase()

        delta_q = current_quality - self.prev_quality
        is_turbulent = False
        if len(self.history) >= 3:
            q_values = [h[0] for h in self.history[-3:]]
            deltas = [q_values[i] - q_values[i - 1] for i in range(1, len(q_values))]
            if all(deltas[i] * deltas[i - 1] < 0 for i in range(1, len(deltas))):
                if all(abs(d) > self.min_delta * 2 for d in deltas):
                    is_turbulent = True

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
                msg_parts.append("[TRAPPED] Fidelity Floor reached. Relaxing stagnation guard.")

        regress_threshold = -0.03 if self.task_type in ["quality", "parameter_prediction"] else -0.01
        is_regressing = delta_q < (self.prev_quality * regress_threshold) if self.prev_quality else False

        if self.task_type == "forex":
            is_collapsed = current_quality < 45.0
        else:
            is_collapsed = (current_quality < 0.05) or (plcc < -0.1)

        loss_is_stable = (current_loss <= self.prev_loss * 1.05) if current_loss and self.prev_loss else True
        is_expanding = phase in ["FOUNDATION", "EXPANSION"]

        if plcc < -0.01 and self.current_temp > 0.7 and (target_std is None or target_std >= 0.15):
            self.current_temp = 0.5
            self.current_clamp = 20.0
            t_changed = c_changed = True
            msg_parts.append("[THERMAL SHOCK] PLCC negative. Sharpening manifold (Temp -> 0.5).")

        if self.spatial_lock_remaining > 0:
            self.spatial_lock_remaining -= 1
            loss_is_exploding = (current_loss > self.prev_loss * 1.25) if current_loss and self.prev_loss else False
            if is_regressing and not loss_is_exploding:
                is_regressing = False
                msg_parts.append(f"[SPATIAL LOCK] Buffering transition (Patience: {self.spatial_lock_remaining})")

        current_state = (self.current_res, round(self.current_fraction, 2))
        failures = self.failure_log.get(str(current_state), 0)

        if self.task_type in ["quality", "forex"]:
            is_turbulent = False
            if is_expanding:
                msg_parts.append("[SIGNAL] [RESONANCE] Turbulence detected but shielded. Holding manifold.")

        should_retreat = is_regressing or is_turbulent or is_collapsed
        if is_expanding and loss_is_stable and not is_collapsed:
            should_retreat = False
            if is_regressing:
                msg_parts.append("[GUARD] [MOMENTUM] Jitter detected but Loss is stable. Holding manifold.")

        if is_overfitting and is_expanding:
            should_retreat = False
            self.cooldown_remaining = 0
            msg_parts.append("[RESCUE] [OVERFITTING] Overfitting detected. Forcing dataset expansion to introduce variety.")
        elif is_overfitting and self.current_fraction >= 1.0:
            should_retreat = False
            self.cooldown_remaining = 0
            if self.task_type == "forex":
                self.current_stress = min(2.0, self.current_stress + 0.5)
                msg_parts.append(f"[REGULARIZE] [OVERFITTING] Temporal fold divergence detected. Applying mild stabilization (Level {self.current_stress}).")
            else:
                self.current_stress = min(5.0, self.current_stress + 1.0)
                msg_parts.append(f"[RESCUE] [OVERFITTING] Dataset exhausted. Deploying Stress Protocol (Level {self.current_stress}).")

        if should_retreat:
            self.failure_log[str(current_state)] = failures + 1
            msg_parts.append(f"[WARNING] NPP FAILURE: State {current_state} (Count: {self.failure_log[str(current_state)]})")

            if self.failure_log[str(current_state)] >= 2:
                self.current_clamp = max(10.0, self.current_clamp - 5.0)
                c_changed = True
                self.current_temp = min(1.8, self.current_temp * 1.5)
                t_changed = True
                msg_parts.append("NPP LOOP: Forcing Numerical Shakeup")

            proven_threshold = max(85.0 if self.task_type == "quality" else 0.85, self.target_quality_score * 0.75)
            is_proven_manifold = self.best_quality >= proven_threshold

            can_spatial_retreat = (
                self.epoch_count - self.last_res_jump_epoch < 8
                and self.breakout_lock == 0
                and not is_proven_manifold
            )

            if can_spatial_retreat and self.current_res in self.res_ladder:
                res_idx = self.res_ladder.index(self.current_res)
                if res_idx > 0:
                    self.current_res = self.res_ladder[res_idx - 1]
                    self.current_fraction = 1.0
                    r_changed = f_changed = True
                    self.lr_multiplier = self.cooling_factor
                    lr_changed = True
                    self.stabilization_epochs = self.stabilization_lock
                    self.cooldown_remaining = 5
                    msg_parts.append(f"[SPATIAL RETREAT] Resetting to {self.current_res}px @ 100% Data Anchor")

            if not r_changed:
                opt_cfg = self.model_info.get("optimization", {})
                initial_frac = opt_cfg.get("initial_fraction", 0.15)
                if self.current_fraction > initial_frac:
                    prev_frac = max(initial_frac, round(self.current_fraction - self.fraction_increment, 2))
                    if prev_frac < self.current_fraction:
                        self.current_fraction = prev_frac
                        f_changed = True
                        msg_parts.append(f"[DATA RECOIL] Stepping back dataset fraction: {self.current_fraction*100:.0f}% @ {self.current_res}px")

                self.lr_multiplier = self.cooling_factor
                lr_changed = True
                self.stabilization_epochs = self.stabilization_lock
                self.cooldown_remaining = 5
                if not f_changed:
                    msg_parts.append(f"RECOIL: Retaining data fraction at {self.current_fraction*100:.0f}% | Cooling LR to stabilize manifold")

        elif sentinel_trigger_rate > 0.15:
            self.current_temp = min(1.2, self.current_temp * 1.2)
            self.lr_multiplier = 0.75
            self.current_stress = max(0.0, self.current_stress - 0.25)
            lr_changed = True
            t_changed = True
            self.stabilization_epochs = 2
            msg_parts.append(f"COOLING: Stress {sentinel_trigger_rate*100:.1f}% -> Temp {self.current_temp:.2f} | DataStress {self.current_stress:.1f}")

        stride_threshold = 0.75 if (self.current_res is not None and self.current_res < 512) else 0.90
        if self.target_quality_score > 1.0:
            stride_threshold = stride_threshold * self.target_quality_score
        propulsion_allowed = not should_retreat and self.cooldown_remaining == 0
        not_regressing = delta_q >= -self.min_delta

        trigger_propulsion = propulsion_allowed and (
            (is_overfitting and is_expanding)
            or is_plateaued
            or (not_regressing and (is_flat_leg or (current_quality > stride_threshold and delta_q < self.min_delta)))
        )

        if self.jolt_window_remaining > 0:
            self.jolt_window_remaining -= 1
            collapse_threshold = (self.prev_quality * -0.03) if (self.prev_quality and self.prev_quality > 1.0) else -0.015
            if delta_q < collapse_threshold or should_retreat or is_regressing:
                self.jolt_window_remaining = 0
                self.lr_multiplier = self.cooling_factor
                self.head_lr_multiplier = self.cooling_factor
                lr_changed = True
                msg_parts.append(f"[JOLT SHIELD] Early collapse triggered (Regression: {delta_q:.4f} < {collapse_threshold:.4f}). Cooling LR.")
            else:
                lr_changed = True
                msg_parts.append(f"SUSTAINED JOLT: Window Active ({self.jolt_window_remaining} epochs remaining | Head LR: {self.head_lr_multiplier:.2f}x | Backbone LR: {self.lr_multiplier:.2f}x)")

        if trigger_propulsion:
            jolt_ready = (self.epoch_count - self.last_jolt_epoch) > self.jolt_cooldown and self.cooldown_remaining == 0
            if is_flat and jolt_ready and self.jolt_window_remaining == 0:
                jolt = self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5)
                if is_trapped:
                    jolt *= 1.5
                if self.task_type == "forex":
                    jolt *= 2.0
                jolt_base = float(jolt)

                if phase == "REFINEMENT":
                    self.lr_multiplier = round(jolt_base * 0.5, 3)
                    self.head_lr_multiplier = min(round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3))
                else:
                    self.lr_multiplier = jolt_base
                    self.head_lr_multiplier = jolt_base

                window_size = 5 if self.task_type == "forex" else 3
                self.jolt_window_remaining = window_size
                lr_changed = True
                self.last_jolt_epoch = self.epoch_count
                msg_parts.append(f"JOLT: Breaking Plateau with Head-Differential Propulsion (Head: {self.head_lr_multiplier:.2f}x, Backbone: {self.lr_multiplier:.2f}x | {window_size}-Epoch Window)")

            next_frac = min(1.0, self.current_fraction + self.fraction_increment)
            next_state = (self.current_res, round(next_frac, 2))

            if self.failure_log.get(str(next_state), 0) > 0:
                self.lr_multiplier = 0.6
                self.head_lr_multiplier = 0.6
                lr_changed = True
                msg_parts.append("ANCHOR: Caution ahead (Previous Failures). 0.6x LR.")

            if phase in ("FOUNDATION", "EXPANSION"):
                if self.current_fraction < 1.0:
                    self.current_fraction = next_frac
                    f_changed = True
                    msg_parts.append(f"PROPULSION: Data -> {self.current_fraction*100:.0f}%")
                    self.stabilization_epochs = 1
                    self.best_quality = current_quality
            elif phase == "DEEPENING" and self.current_res is not None and self.current_res in self.res_ladder:
                current_idx = self.res_ladder.index(int(self.current_res))
                next_res = self.res_ladder[current_idx + 1]

                if self.best_quality < self.target_quality_score * 0.80 and self.target_quality_score > 1.0:
                    self.lr_multiplier = self.cooling_factor
                    self.head_lr_multiplier = self.cooling_factor
                    lr_changed = True
                    msg_parts.append("RECOIL: Insufficient Quality for spatial jump. Cooling LR.")
                    self.stabilization_epochs = self.stabilization_lock
                else:
                    self.current_res = next_res
                    r_changed = b_changed = True
                    self.current_fraction = 0.15
                    f_changed = True
                    self.last_res_jump_epoch = self.epoch_count
                    self.spatial_lock_remaining = self.stabilization_lock
                    msg_parts.append(f"SPATIAL JUMP: {next_res}px | Data Reset 15% | Lock: ON")
                    self.stabilization_epochs = self.stabilization_lock
            else:
                if self.task_type == "forex":
                    if self.current_stress < 2.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                        self.current_stress = min(2.0, self.current_stress + 0.5)
                        self.lr_multiplier = 1.0
                        self.head_lr_multiplier = 1.10
                        self.jolt_window_remaining = 2
                        lr_changed = True
                        msg_parts.append("REFINEMENT: Plateau in Temporal Fold. Deploying Gentle Fold Tuning (Head LR: 1.10x)")
                    else:
                        if self.jolt_window_remaining == 0:
                            self.lr_multiplier = self.cooling_factor
                            self.head_lr_multiplier = self.cooling_factor
                            lr_changed = True
                            msg_parts.append("REFINEMENT: Precision Cooling")
                elif self.current_stress < 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    self.current_stress = min(5.0, self.current_stress + 1.0)
                    jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    self.lr_multiplier = round(jolt_base * 0.5, 3)
                    self.head_lr_multiplier = min(round(self.lr_multiplier * 3.0, 3), round(jolt_base * 1.5, 3))
                    self.jolt_window_remaining = 3
                    lr_changed = True
                    self.max_stress_stuck_epochs = 0
                    msg_parts.append(f"REFINEMENT: Trapped in Plateau. Deploying Stress Protocol (Level {self.current_stress}) & Differential Jolt")
                elif self.current_stress >= 5.0 and self.target_quality_score > 0 and self.best_quality < self.target_quality_score * 0.90:
                    jolt_base = float(self.model_info.get("optimization", {}).get("jolt_multiplier", 1.5))
                    self.lr_multiplier = jolt_base
                    self.head_lr_multiplier = jolt_base
                    lr_changed = True
                    self.max_stress_stuck_epochs = self.max_stress_stuck_epochs + 1
                    stuck_patience = self.plateau_patience * 2
                    msg_parts.append(f"REFINEMENT: [MAX STRESS] Forcing Jolt (x{jolt_base:.2f}) to maintain momentum (Stuck: {self.max_stress_stuck_epochs}/{stuck_patience})")
                    if self.max_stress_stuck_epochs >= stuck_patience:
                        self.trigger_mini_swa = True
                        msg_parts.append(f"[MINI-SWA PULSE] Triggering weight averaging pulse across top checkpoints (Stuck: {self.max_stress_stuck_epochs} epochs)")
                else:
                    if self.jolt_window_remaining == 0:
                        self.lr_multiplier = self.cooling_factor
                        self.head_lr_multiplier = self.cooling_factor
                        lr_changed = True
                        msg_parts.append("REFINEMENT: SOTA Precision Cooling")

                    if metrics_dict is None:
                        metrics_dict = {"plcc": plcc, "srcc": srcc}

                    lagging = self._identify_lagging_metric(metrics_dict)
                    if lagging:
                        adjustments = self._apply_targeted_optimizations(lagging, metrics_dict)
                        msg_parts.extend(adjustments)

        if not (is_regressing or is_turbulent or sentinel_trigger_rate > 0.15) and self.current_temp > self.min_temp:
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
                t_changed = True
                msg_parts.append(f"SHARPENING: Temp -> {self.current_temp:.2f}")
            elif self.cooldown_remaining > 0:
                msg_parts.append(f"MEDITATION: Cooldown active ({self.cooldown_remaining} epochs)")

        self.prev_quality = current_quality
        if current_loss:
            self.prev_loss = current_loss

        if self.task_type == "quality":
            if self.cooldown_remaining == 0:
                self.current_temp = min(1.0, self.current_temp)
            self.current_clamp = min(self.clamp_range[1], self.current_clamp)

        early_stop_triggered = False
        if epochs_no_improve >= self.plateau_patience:
            if self.metric_focus_epochs_remaining == 0 and self.stabilization_epochs == 0 and self.cooldown_remaining == 0:
                early_stop_triggered = True
                msg_parts.append(f"[EARLY STOPPING] Patience exceeded ({epochs_no_improve}/{self.plateau_patience}). Fold exhausted.")

        final_msg = f"[LAUNCH] [{phase}] " + " | ".join(msg_parts) if msg_parts else ""

        return (
            f_changed,
            r_changed,
            lr_changed,
            t_changed,
            c_changed,
            b_changed,
            early_stop_triggered,
            final_msg,
        )

    def get_dynamic_save_interval(self, avg_iter_time: float, total_iters: int) -> float:
        """Calculate dynamic checkpoint frequency based on iteration time."""
        if avg_iter_time <= 0:
            return 0.2
        secs_per_epoch = avg_iter_time * total_iters
        if secs_per_epoch > 7200:
            target_pct = 900.0 / secs_per_epoch
        elif secs_per_epoch > 3600:
            target_pct = 1200.0 / secs_per_epoch
        else:
            return 0.5
        return max(0.15, min(0.5, target_pct))

    def veto_resolution_jump(self, fallback_res: int, reason: str = "VRAM physical ceiling") -> None:
        """Lock training to fallback_res and prevent higher resolution attempts."""
        self.current_res = fallback_res
        self.spatial_lock_remaining = 10
        self.stabilization_epochs = 10
        if self.res_ladder:
            self.res_ladder = [r for r in self.res_ladder if r <= fallback_res]
            if not self.res_ladder:
                self.res_ladder = [fallback_res]
        logger.info(
            "Spatial jump VETOED (%s). Anchoring at %dpx (Lock: ON).",
            reason,
            fallback_res,
        )

    def recoil(self) -> str:
        """Recoil to softer numerical conditions on regression."""
        self.current_temp = min(1.5, self.current_temp * 1.3)
        if self.task_type == "quality":
            self.current_temp = min(1.0, self.current_temp)
        self.stabilization_epochs = 3
        return f"[NPP] RECOIL: Retaining data fraction at {self.current_fraction*100:.0f}% | Temp Heatup {self.current_temp:.2f}"

    def reset_best(self) -> None:
        """Reset historical SOTA tracking."""
        self.sota.reset_best()

    def register_rollback(self) -> None:
        """Register a model rollback and execute loop breaking logic."""
        if not self.loop_breaker_enabled:
            return

        res_before = self.current_res
        result = self.sota.register_rollback(self.current_res, self.res_ladder)

        if result.get("breakout_triggered"):
            new_res = result.get("new_res")
            if new_res is not None:
                self.current_res = new_res
                if self.current_res not in self.res_ladder:
                    self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
                self.current_fraction = 0.15
                logger.info(
                    "Breakout promoted resolution %dpx -> %dpx with retreat protection.",
                    res_before,
                    self.current_res,
                )

    def get_active_drift_gate(self, config_gate: float) -> float:
        """Retrieve active drift gate considering relaxation."""
        return self.sota.get_active_drift_gate(config_gate)

    def get_active_regression_limit(self, config_limit: int) -> int:
        """Retrieve active regression limit considering rollback history."""
        return self.sota.get_active_regression_limit(config_limit, self.current_res)

    def get_state(self) -> dict[str, Any]:
        """Serialize complete governor state for checkpoint persistence."""
        return {
            "sample_fraction": self.current_fraction,
            "input_size": self.current_res,
            "softmax_temp": self.current_temp,
            "logit_clamp": self.current_clamp,
            "lr_multiplier": self.lr_multiplier,
            "head_lr_multiplier": self.head_lr_multiplier,
            "jolt_window_remaining": self.jolt_window_remaining,
            "batch_size": self.current_batch,
            "accumulation_steps": self.current_acc,
            "stabilization_epochs": self.stabilization_epochs,
            "failure_log": dict(self.failure_log),
            "history": list(self.history),
            "thermal_floor": dict(self.thermal_floor),
            "cooldown_remaining": self.cooldown_remaining,
            "spatial_lock_remaining": self.spatial_lock_remaining,
            "last_res_jump_epoch": self.last_res_jump_epoch,
            "epoch_count": self.epoch_count,
            "best_quality": self.best_quality,
            "stress": self.current_stress,
            "last_jolt_epoch": self.last_jolt_epoch,
            "max_stress_stuck_epochs": self.max_stress_stuck_epochs,
            "consecutive_rollbacks": self.consecutive_rollbacks,
            "rollback_history": dict(self.rollback_history),
            "gate_relaxation_epochs": self.gate_relaxation_epochs,
            "sota_resolution": self.sota_resolution,
            "breakout_lock": self.breakout_lock,
            "rank_weight": self.current_rank_weight,
            "rank_margin": self.current_rank_margin,
            "soft_spearman_weight": self.current_spearman_weight,
            "lpips_weight": self.current_lpips_weight,
            "mag_weight": self.current_mag_weight,
            "dir_weight": self.current_dir_weight,
            "emd_weight": self.current_emd_weight,
            "ssim_weight": self.current_ssim_weight,
            "huber_delta": self.current_huber_delta,
            "conf_gate_str": self.current_conf_gate_str,
            "metric_focus_epochs_remaining": self.metric_focus_epochs_remaining,
            "metric_focus_target": self.metric_focus_target,
            "metric_focus_last_fired": dict(self.metric_focus_last_fired),
        }

    def load_state(self, state: dict[str, Any] | None, preserve_curriculum: bool = False) -> None:
        """Restore governor state from checkpoint dictionary."""
        if not state:
            return

        if not preserve_curriculum:
            self.current_fraction = float(state.get("sample_fraction", self.current_fraction))
            raw_res = state.get("input_size", self.current_res)
            self.current_res = int(raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res)

            if self.task_type == "forex" and self.epoch_count <= 2:
                opt_cfg = self.model_info.get("optimization", {})
                init_frac = float(opt_cfg.get("initial_fraction", 0.15))
                saved_frac = float(state.get("sample_fraction", 1.0))
                if saved_frac >= 0.99 and init_frac < 0.99:
                    logger.info("Overriding legacy 100% fraction from checkpoint with configured: %.1f%%.", init_frac * 100.0)
                    self.current_fraction = init_frac

            if self.current_res is not None and self.current_res not in self.res_ladder:
                self.res_ladder = sorted(list(set(self.res_ladder + [self.current_res])))
        else:
            raw_res = state.get("input_size", self.current_res)
            self.sota_resolution = int(raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res)

        self.current_temp = max(self.min_temp, float(state.get("softmax_temp", self.current_temp)))
        if self.task_type == "quality":
            self.current_temp = min(1.0, self.current_temp)
        self.current_clamp = float(state.get("logit_clamp", self.current_clamp))

        if "rank_weight" in state:
            self.current_rank_weight = float(state["rank_weight"])
        if "rank_margin" in state:
            self.current_rank_margin = float(state["rank_margin"])
        if "soft_spearman_weight" in state:
            self.current_spearman_weight = float(state["soft_spearman_weight"])
        if "lpips_weight" in state:
            self.current_lpips_weight = float(state["lpips_weight"])
        if "mag_weight" in state:
            self.current_mag_weight = float(state["mag_weight"])
        if "dir_weight" in state:
            self.current_dir_weight = float(state["dir_weight"])
        if "emd_weight" in state:
            self.current_emd_weight = float(state["emd_weight"])
        if "ssim_weight" in state:
            self.current_ssim_weight = float(state["ssim_weight"])
        if "huber_delta" in state:
            self.current_huber_delta = float(state["huber_delta"])
        if "conf_gate_str" in state:
            self.current_conf_gate_str = float(state["conf_gate_str"])

        self.metric_focus_epochs_remaining = int(state.get("metric_focus_epochs_remaining", 0))
        self.metric_focus_target = state.get("metric_focus_target")
        raw_mflf = state.get("metric_focus_last_fired", {})
        self.metric_focus_last_fired = {k: int(v) for k, v in raw_mflf.items()}

        self.lr_multiplier = float(state.get("lr_multiplier", self.lr_multiplier))
        self.head_lr_multiplier = float(state.get("head_lr_multiplier", self.lr_multiplier))
        self.jolt_window_remaining = int(state.get("jolt_window_remaining", 0))
        self.current_batch = int(state.get("batch_size", self.current_batch))
        self.current_acc = int(state.get("accumulation_steps", self.current_acc))
        self.stabilization_epochs = int(state.get("stabilization_epochs", 0))
        self.failure_log = dict(state.get("failure_log", {}))
        self.history = list(state.get("history", []))
        self.thermal_floor = dict(state.get("thermal_floor", {}))
        self.cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self.last_jolt_epoch = int(state.get("last_jolt_epoch", -10))
        self.spatial_lock_remaining = int(state.get("spatial_lock_remaining", 0))
        self.last_res_jump_epoch = int(state.get("last_res_jump_epoch", 0))
        self.epoch_count = int(state.get("epoch_count", self.epoch_count))
        self.best_quality = float(state.get("best_quality", self.best_quality))
        self.current_stress = float(state.get("stress", 0.0))
        self.max_stress_stuck_epochs = int(state.get("max_stress_stuck_epochs", 0))

        self.consecutive_rollbacks = int(state.get("consecutive_rollbacks", 0))
        raw_history = state.get("rollback_history", {})
        self.rollback_history = {int(k): int(v) for k, v in raw_history.items()}
        self.gate_relaxation_epochs = int(state.get("gate_relaxation_epochs", 0))
        self.sota_resolution = state.get("sota_resolution")
        self.breakout_lock = int(state.get("breakout_lock", 0))
