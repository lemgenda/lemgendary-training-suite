"""Checkpoint serialisation of the governor's curriculum, thermal and SOTA state."""

from __future__ import annotations

from typing import Any

from training.governance.governor_host import GovernorHost
from training.ratchet import (
    FRACTION_STATE_KEY,
    RESOLUTION_STATE_KEY,
)


class GovernorStateMixin(GovernorHost):
    """Checkpoint serialisation of the governor's curriculum, thermal and SOTA state."""

    def get_state(self) -> dict[str, Any]:
        """Serialize complete governor state for checkpoint persistence."""
        return {
            FRACTION_STATE_KEY: self.current_fraction,
            RESOLUTION_STATE_KEY: self.current_res,
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
            "last_action_epoch": self.last_action_epoch,
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

        # Restore epoch counters and history first so curriculum checks have accurate state context
        self.epoch_count = int(state.get("epoch_count", self.epoch_count))
        self.last_res_jump_epoch = int(state.get("last_res_jump_epoch", 0))
        self.last_action_epoch = int(state.get("last_action_epoch", self.last_res_jump_epoch))
        self.history = list(state.get("history", []))

        if not preserve_curriculum:
            raw_frac = float(state.get(FRACTION_STATE_KEY, self.current_fraction))
            self.current_fraction = 1.0 if (1.0 - raw_frac < 0.01) else round(raw_frac, 4)
            raw_res = state.get(RESOLUTION_STATE_KEY, self.current_res)
            self.current_res = int(raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res)

            # The saved fraction is restored as-is: lowering it at the same
            # resolution would break the Monotonic Ratchet (pair rule).
            if (
                self.task_type in ("forex", "llm", "text_instruction")
                or self.model_info.get("category") == "llm"
            ):
                self.current_fraction = 1.0

            if self.current_res is not None and self.current_res not in self.res_ladder:
                self.res_ladder = sorted(set(self.res_ladder + [self.current_res]))
        else:
            raw_res = state.get(RESOLUTION_STATE_KEY, self.current_res)
            self.sota_resolution = int(
                raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res
            )

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
        self.thermal_floor = dict(state.get("thermal_floor", {}))
        self.cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self.last_jolt_epoch = int(state.get("last_jolt_epoch", -10))
        self.spatial_lock_remaining = int(state.get("spatial_lock_remaining", 0))
        self.best_quality = float(state.get("best_quality", self.best_quality))
        self.current_stress = float(state.get("stress", 0.0))
        self.max_stress_stuck_epochs = int(state.get("max_stress_stuck_epochs", 0))

        self.consecutive_rollbacks = int(state.get("consecutive_rollbacks", 0))
        raw_history = state.get("rollback_history", {})
        self.rollback_history = {int(k): int(v) for k, v in raw_history.items()}
        self.gate_relaxation_epochs = int(state.get("gate_relaxation_epochs", 0))
        self.sota_resolution = state.get("sota_resolution")
        self.breakout_lock = int(state.get("breakout_lock", 0))
