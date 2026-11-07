"""Central governor coordinating curriculum, thermal, metrics, and SOTA tracking."""

from __future__ import annotations

import logging
from typing import Any

from training.data.workers import release_cuda_memory
from training.governance.epoch_audit import EpochAuditMixin
from training.governance.governor_groups import GovernorSpec
from training.governance.governor_setup import (
    build_curriculum,
    build_runtime,
    build_sota,
    build_thermal,
    build_tuning,
    resolve_task_type,
)
from training.governance.governor_state import GovernorStateMixin
from training.governance.metrics import MetricRegistry
from training.governance.propulsion import PropulsionMixin
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
    and per-rung SOTA stability management. Note that ``self.sota`` tracks per-rung
    curriculum plateau/soak state and may reset across resolution jumps, whereas the
    training engine's top-level ``ctx.governance.sota_tracker`` tracks the global
    all-time best Quality Score used to gate ``best.pth`` persistence.
    """

    def __init__(
        self,
        model_info: dict[str, Any],
        config: dict[str, Any] | None = None,
        stabilizers: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}
        stab = stabilizers or {}
        opt = model_info.get("optimization", {})
        task_type = resolve_task_type(model_info)
        self.spec = GovernorSpec(
            model_info=model_info,
            config=config,
            stab=stab,
            enabled=opt.get("enabled", True),
            task_type=task_type,
            sota_targets=opt.get("sota_targets", model_info.get("sota_targets", {})),
        )
        self.metrics = MetricRegistry()
        self.curriculum = build_curriculum(model_info, config, task_type)
        self.tuning = build_tuning(model_info, config, task_type)
        self.thermal = build_thermal(stab, task_type, self.tuning.clamp_range)
        self.runtime = build_runtime(stab, self.thermal.min_temperature)
        self.sota = build_sota(model_info, config, task_type, self.spec.sota_targets)

    # NOTE: The signature must match EpochAuditMixin._commit_rung (int, float) → None.
    # The original parameter names were ``new_res`` and ``new_frac``; we keep them
    # internally for readability while exposing the exact expected signature.
    def _commit_rung(self, rung: int, res: float) -> None:
        """Move the curriculum to ``(rung, res)`` under the pair rule.

        Raises ``RatchetViolation`` if the move would lower the resolution, or
        lower the fraction without a resolution step‑up.
        """
        # Preserve the original variable names for internal clarity.
        new_res, new_frac = rung, res

        check_transition(
            Rung(self.current_res, float(self.current_fraction)),
            Rung(new_res, float(new_frac)),
        )
        self.curriculum.current_res = new_res
        self.curriculum.current_fraction = float(new_frac)

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
        release_cuda_memory("OOM handling")

        if current_batch is not None and current_batch > 0:
            self.curriculum.current_batch = current_batch

        new_batch, new_acc = self.curriculum.throttle_on_oom(min_batch=min_batch)
        self.runtime.counters.spatial_lock_remaining = max(self.spatial_lock_remaining, 5)
        self.sota.stabilization_epochs = max(self.stabilization_epochs, 3)

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

    def recoil(self) -> str:
        """Recoil to softer numerical conditions on regression."""
        self.thermal.temperature = min(1.5, self.current_temp * 1.3)
        if self.task_type == "quality":
            self.thermal.temperature = min(1.0, self.current_temp)
        self.sota.stabilization_epochs = 3
        return (
            f"[NPP] RECOIL: Retaining data fraction at {self.current_fraction * 100:.0f}% | "
            f"Temp Heatup {self.current_temp:.2f}"
        )
