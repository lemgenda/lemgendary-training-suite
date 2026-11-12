"""Checkpoint serialisation of the governor's curriculum, thermal and SOTA state."""

from __future__ import annotations

import logging
from typing import Any

from training.governance.governor_host import GovernorHost
from training.governance.quality_score import (
    QUALITY_SCHEME_KEY,
    anchor_is_comparable,
    quality_scheme,
)
from training.ratchet import (
    FRACTION_STATE_KEY,
    RESOLUTION_STATE_KEY,
)

logger = logging.getLogger("lemtrain.governance.governor")

# Checkpoint key -> governor attribute of the loss weights restored only when saved.
_OPTIONAL_WEIGHT_KEYS: dict[str, str] = {
    "rank_weight": "current_rank_weight",
    "rank_margin": "current_rank_margin",
    "soft_spearman_weight": "current_spearman_weight",
    "lpips_weight": "current_lpips_weight",
    "mag_weight": "current_mag_weight",
    "dir_weight": "current_dir_weight",
    "emd_weight": "current_emd_weight",
    "ssim_weight": "current_ssim_weight",
    "huber_delta": "current_huber_delta",
    "conf_gate_str": "current_conf_gate_str",
}


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
            QUALITY_SCHEME_KEY: quality_scheme(self.task_type),
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
        counters = self.runtime.counters
        counters.epoch_count = int(state.get("epoch_count", counters.epoch_count))
        counters.last_res_jump_epoch = int(state.get("last_res_jump_epoch", 0))
        counters.last_action_epoch = int(
            state.get("last_action_epoch", counters.last_res_jump_epoch)
        )
        comparable = anchor_is_comparable(state.get(QUALITY_SCHEME_KEY), self.task_type)
        self.sota.history = list(state.get("history", [])) if comparable else []

        if not preserve_curriculum:
            self._restore_rung(state)
        else:
            self.sota.sota_resolution = _resolution(
                state.get(RESOLUTION_STATE_KEY, self.current_res)
            )

        self._restore_loss_shaping(state)
        self._restore_windows(state)
        self._restore_sota(state, restore_best=comparable)
        if not comparable:
            logger.warning(
                "Checkpoint quality scores use another scheme (%s, now %s); the best quality "
                "%.2f and the quality history are not reused and are rebuilt from new epochs.",
                state.get(QUALITY_SCHEME_KEY, "untagged"),
                quality_scheme(self.task_type),
                float(state.get("best_quality", 0.0) or 0.0),
            )

    def _restore_rung(self, state: dict[str, Any]) -> None:
        """Restore the saved (resolution, fraction) rung and keep it on the ladder."""
        curriculum = self.curriculum
        raw_frac = float(state.get(FRACTION_STATE_KEY, curriculum.current_fraction))
        curriculum.current_fraction = 1.0 if (1.0 - raw_frac < 0.01) else round(raw_frac, 4)
        curriculum.current_res = _resolution(
            state.get(RESOLUTION_STATE_KEY, curriculum.current_res)
        )

        # The saved fraction is restored as-is: lowering it at the same
        # resolution would break the Monotonic Ratchet (pair rule).
        if (
            self.task_type in ("forex", "llm", "text_instruction")
            or self.model_info.get("category") == "llm"
        ):
            curriculum.current_fraction = 1.0

        if (
            curriculum.current_res is not None
            and curriculum.current_res not in curriculum.res_ladder
        ):
            curriculum.res_ladder = sorted(set(curriculum.res_ladder + [curriculum.current_res]))

    def _restore_loss_shaping(self, state: dict[str, Any]) -> None:
        """Restore temperature, clamp and the loss weights present in the checkpoint."""
        self.thermal.temperature = float(
            max(self.min_temp, float(state.get("softmax_temp", self.current_temp)))
        )
        if self.task_type == "quality":
            self.thermal.temperature = min(1.0, self.thermal.temperature)
        calibration = self.runtime.calibration
        calibration.current_clamp = float(state.get("logit_clamp", calibration.current_clamp))
        calibration.thermal_floor = dict(state.get("thermal_floor", {}))

        for key, attr in _OPTIONAL_WEIGHT_KEYS.items():
            if key in state:
                setattr(self, attr, float(state[key]))

    def _restore_windows(self, state: dict[str, Any]) -> None:
        """Restore the metric-focus window, LR multipliers, batch and per-epoch windows."""
        focus = self.runtime.focus
        focus.metric_focus_epochs_remaining = int(state.get("metric_focus_epochs_remaining", 0))
        focus.metric_focus_target = state.get("metric_focus_target")
        raw_mflf = state.get("metric_focus_last_fired", {})
        focus.metric_focus_last_fired = {k: int(v) for k, v in raw_mflf.items()}

        lr = self.runtime.lr
        lr.lr_multiplier = float(state.get("lr_multiplier", lr.lr_multiplier))
        lr.head_lr_multiplier = float(state.get("head_lr_multiplier", lr.lr_multiplier))
        lr.jolt_window_remaining = int(state.get("jolt_window_remaining", 0))
        self.curriculum.current_batch = int(state.get("batch_size", self.curriculum.current_batch))
        self.curriculum.current_acc = int(
            state.get("accumulation_steps", self.curriculum.current_acc)
        )
        counters = self.runtime.counters
        counters.last_jolt_epoch = int(state.get("last_jolt_epoch", -10))
        counters.spatial_lock_remaining = int(state.get("spatial_lock_remaining", 0))

    def _restore_sota(self, state: dict[str, Any], restore_best: bool = True) -> None:
        """Restore stress, cooldown, best quality and the loop-breaker rollback state.

        ``restore_best`` is False when the saved best quality is on another score scale.
        """
        stress = self.sota.stress
        stress.stabilization_epochs = int(state.get("stabilization_epochs", 0))
        stress.failure_log = dict(state.get("failure_log", {}))
        stress.cooldown_remaining = int(state.get("cooldown_remaining", 0))
        stress.current_stress = float(state.get("stress", 0.0))
        stress.max_stress_stuck_epochs = int(state.get("max_stress_stuck_epochs", 0))
        if restore_best:
            self.sota.quality.best_quality = float(state.get("best_quality", self.best_quality))

        rollback = self.sota.rollback
        rollback.consecutive_rollbacks = int(state.get("consecutive_rollbacks", 0))
        raw_history = state.get("rollback_history", {})
        rollback.rollback_history = {int(k): int(v) for k, v in raw_history.items()}
        rollback.gate_relaxation_epochs = int(state.get("gate_relaxation_epochs", 0))
        rollback.sota_resolution = state.get("sota_resolution")
        rollback.breakout_lock = int(state.get("breakout_lock", 0))


def _resolution(raw_res: Any) -> int:
    """Return the resolution of a saved size (height of a pair)."""
    return int(raw_res[1] if isinstance(raw_res, (list, tuple)) else raw_res)
