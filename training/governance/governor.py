"""Central governor coordinating curriculum, thermal, metrics, and SOTA tracking."""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from training.governance.curriculum import CurriculumState
from training.governance.epoch_audit import EpochAuditMixin
from training.governance.governor_state import GovernorStateMixin
from training.governance.metrics import (
    DEFAULT_METRIC_DIRECTIONS,
    DEFAULT_METRIC_WEIGHTS,
    MetricRegistry,
)
from training.governance.propulsion import PropulsionMixin
from training.governance.sota import SotaTracker
from training.governance.thermal import ThermalState
from training.ratchet import (
    Rung,
    check_transition,
)

logger = logging.getLogger("lemtrain.governance.governor")


class GovernorStateError(Exception):
    """Raised when an invalid state transition or operation occurs in the governor."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class SmartTrainingGovernor(EpochAuditMixin, PropulsionMixin, GovernorStateMixin):
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
            manifold_defaults.get("maturity_soak", 3),
        )
        self.sota.plateau_patience = opt.get(
            "plateau_patience",
            self.config.get("governor", {}).get("plateau_patience", 6),
        )

        self.plateau_priority: str = opt.get("plateau_priority", "data")
        self.curriculum.fraction_increment = opt.get(
            "fraction_increment",
            manifold_defaults.get("fraction_increment", 0.20),
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
        is_forex_model = (
            model_info.get("category") == "forex"
            or model_info.get("task_type") == "forex"
            or model_info.get("dataset_type") == "forex"
            or model_info.get("filename") == "ForexPredictor"
        )
        is_llm_model = (
            model_info.get("category") == "llm"
            or model_info.get("task_type") in ("text_instruction", "llm")
            or model_info.get("dataset_type") in ("text_instruction", "llm")
        )
        if is_forex_model or is_llm_model:
            initial_fraction = 1.0
        else:
            initial_fraction = model_info.get("sample_fraction") or opt.get(
                "initial_fraction",
                manifold_defaults.get("initial_fraction", 0.15),
            )
        self.curriculum.current_fraction = float(initial_fraction)

        batch_val = model_info.get("batch_size", 16)
        self.curriculum.current_batch = int(batch_val) if batch_val and batch_val != "auto" else 16
        self.curriculum.current_acc = 1

        raw_size = model_info.get("input_size") or 224
        current_res = raw_size[1] if isinstance(raw_size, (list, tuple)) else raw_size
        try:
            self.curriculum.current_res = int(current_res) if current_res is not None else 224
        except (ValueError, TypeError):
            self.curriculum.current_res = 224

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

        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 8.0
        )
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

            self.curriculum.res_ladder = [
                r for r in self.curriculum.res_ladder if r <= max_safe_res
            ]
            if not self.curriculum.res_ladder:
                self.curriculum.res_ladder = [max_safe_res]
            if (
                self.curriculum.current_res is not None
                and self.curriculum.current_res > max_safe_res
            ):
                logger.info(
                    "Hardware VRAM Cap Active (%.1fGB): Clamping %dpx -> %dpx for memory "
                    "stability.",
                    vram_gb,
                    self.curriculum.current_res,
                    max_safe_res,
                )
                self.curriculum.current_res = max_safe_res

        if (
            self.curriculum.res_ladder
            and self.curriculum.current_res != self.curriculum.res_ladder[0]
            and self.task_type != "forex"
        ):
            logger.info(
                "Aligning start resolution %dpx -> lowest rung %dpx.",
                self.curriculum.current_res,
                self.curriculum.res_ladder[0],
            )
            self.curriculum.current_res = self.curriculum.res_ladder[0]

        if (
            self.curriculum.current_res is not None
            and self.curriculum.current_res not in self.curriculum.res_ladder
        ):
            self.curriculum.res_ladder = sorted(
                set(self.curriculum.res_ladder + [self.curriculum.current_res])
            )

        if self.task_type == "forex":
            self.min_temp = max(0.75, float(self.stab.get("min_temp", 0.75)))
            self.current_temp = max(self.min_temp, float(self.stab.get("softmax_temp", 1.0)))
        else:
            default_min = (
                0.5
                if self.task_type == "quality"
                else (0.01 if self.task_type == "parameter_prediction" else 0.1)
            )
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

        self.sota_targets: dict[str, float] = opt.get(
            "sota_targets", model_info.get("sota_targets", {})
        )
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

    def _commit_rung(self, new_res: int, new_frac: float) -> None:
        """Move the curriculum to ``(new_res, new_frac)`` under the pair rule.

        Raises ``RatchetViolation`` if the move would lower the resolution, or
        lower the fraction without a resolution step-up.
        """
        check_transition(
            Rung(int(self.current_res), float(self.current_fraction)),
            Rung(int(new_res), float(new_frac)),
        )
        self.current_res = int(new_res)
        self.current_fraction = float(new_frac)

    # -------------------------------------------------------------------------
    # Core Autonomous Operations
    # -------------------------------------------------------------------------
    def handle_oom(self, current_batch: int | None = None, min_batch: int = 1) -> tuple[int, int]:
        """Governor active on-the-fly OOM recovery intervention.

        1. Immediately purges CUDA memory cache and triggers garbage collection.
        2. Throttles physical batch size and doubles accumulation steps to preserve effective batch.
        3. Locks spatial resolution ladder to prevent premature resolution jumps.
        4. Logs actionable diagnostic telemetry.

        Returns:
            Tuple of (new_batch, new_accumulation_steps).
        """
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                logger.debug("CUDA cache release failed during OOM handling: %s", exc)

        if current_batch is not None and current_batch > 0:
            self.curriculum.current_batch = current_batch

        new_batch, new_acc = self.curriculum.throttle_on_oom(min_batch=min_batch)
        self.spatial_lock_remaining = max(self.spatial_lock_remaining, 5)
        self.stabilization_epochs = max(self.stabilization_epochs, 3)

        logger.warning(
            "[GOVERNOR] [ACTIVE OOM INTERVENTION] VRAM physical ceiling encountered. "
            "Downscaled physical batch to %d, adjusted accumulation to %d. "
            "Spatial jump locked for %d epochs.",
            new_batch,
            new_acc,
            self.spatial_lock_remaining,
        )
        return new_batch, new_acc

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

    def veto_resolution_jump(
        self, fallback_res: int, reason: str = "VRAM physical ceiling"
    ) -> None:
        """Anchor training at fallback_res and prevent higher resolution attempts.

        The veto stops a pending jump; it never retreats. A fallback below the
        current resolution raises ``RatchetViolation`` (Monotonic Ratchet).
        """
        self._commit_rung(fallback_res, self.current_fraction)
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
        return (
            f"[NPP] RECOIL: Retaining data fraction at {self.current_fraction * 100:.0f}% | "
            f"Temp Heatup {self.current_temp:.2f}"
        )

    def register_rollback(self) -> None:
        """Register a model rollback and execute loop breaking logic."""
        if not self.loop_breaker_enabled:
            return

        res_before = self.current_res
        result = self.sota.register_rollback(self.current_res, self.res_ladder)

        if result.get("breakout_triggered"):
            new_res = result.get("new_res")
            if new_res is not None:
                self._commit_rung(new_res, 0.15)
                if self.current_res not in self.res_ladder:
                    self.res_ladder = sorted(set(self.res_ladder + [self.current_res]))
                logger.info(
                    "Breakout promoted resolution %dpx -> %dpx with retreat protection.",
                    res_before,
                    self.current_res,
                )
