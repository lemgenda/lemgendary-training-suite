"""Typed resume state, scheduler runway stretching, and iteration progress scaling."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lemtrain.checkpoint.resume")


@dataclass
class ResumeProgress:
    """Where in the epoch the checkpoint was taken."""

    iteration: int
    is_intra_epoch: bool = False
    phase: str = "train"


@dataclass
class ResumeOptimStates:
    """Optimizer, LR scheduler and AMP scaler state dicts (``None`` when absent)."""

    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None


@dataclass
class ResumeGovernance:
    """Governor snapshot and quality history."""

    governor_state: dict[str, Any] = field(default_factory=dict)
    sota_achieved: bool = False
    best_score: float | None = None
    metric_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResumeAccumulators:
    """Partial-epoch metrics and running accumulators for train and validation."""

    train_metrics: dict[str, Any] = field(default_factory=dict)
    train_accumulators: dict[str, Any] = field(default_factory=dict)
    val_accumulators: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeState:
    """Typed container encapsulating all recoverable state from a training checkpoint.

    The grouped fields stay readable flat (``state.iteration``, ``state.train_metrics``...).
    """

    epoch: int
    model_state_dict: dict[str, Any]
    progress: ResumeProgress
    optim: ResumeOptimStates = field(default_factory=ResumeOptimStates)
    governance: ResumeGovernance = field(default_factory=ResumeGovernance)
    accumulators: ResumeAccumulators = field(default_factory=ResumeAccumulators)

    @property
    def iteration(self) -> int:
        """``progress.iteration``."""
        return self.progress.iteration

    @property
    def is_intra_epoch(self) -> bool:
        """``progress.is_intra_epoch``."""
        return self.progress.is_intra_epoch

    @property
    def phase(self) -> str:
        """``progress.phase``."""
        return self.progress.phase

    @property
    def optimizer_state_dict(self) -> dict[str, Any] | None:
        """``optim.optimizer_state_dict``."""
        return self.optim.optimizer_state_dict

    @property
    def scheduler_state_dict(self) -> dict[str, Any] | None:
        """``optim.scheduler_state_dict``."""
        return self.optim.scheduler_state_dict

    @property
    def scaler_state_dict(self) -> dict[str, Any] | None:
        """``optim.scaler_state_dict``."""
        return self.optim.scaler_state_dict

    @property
    def governor_state(self) -> dict[str, Any]:
        """``governance.governor_state``."""
        return self.governance.governor_state

    @property
    def sota_achieved(self) -> bool:
        """``governance.sota_achieved``."""
        return self.governance.sota_achieved

    @property
    def best_score(self) -> float | None:
        """``governance.best_score``."""
        return self.governance.best_score

    @property
    def metric_history(self) -> list[dict[str, Any]]:
        """``governance.metric_history``."""
        return self.governance.metric_history

    @property
    def train_metrics(self) -> dict[str, Any]:
        """``accumulators.train_metrics``."""
        return self.accumulators.train_metrics

    @property
    def train_accumulators(self) -> dict[str, Any]:
        """``accumulators.train_accumulators``."""
        return self.accumulators.train_accumulators

    @property
    def val_accumulators(self) -> dict[str, Any]:
        """``accumulators.val_accumulators``."""
        return self.accumulators.val_accumulators


def _model_state(payload: dict[str, Any]) -> dict[str, Any]:
    """The weights dict under a known key, or the whole payload when it is raw weights."""
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload


def _progress(payload: dict[str, Any], iteration: int) -> tuple[str, bool, int]:
    """``(phase, is_intra_epoch, iteration)`` from ``progress_metadata`` or top-level keys."""
    progress_meta = payload.get("progress_metadata")
    if isinstance(progress_meta, dict):
        return (
            str(progress_meta.get("phase", "train")),
            True,
            int(progress_meta.get("step", iteration)),
        )
    phase = str(payload.get("phase", "train"))
    return phase, bool(payload.get("phase") == "train" and "step" in payload), iteration


def _first(payload: dict[str, Any], *keys: str) -> Any:
    """``payload.get(k1) or payload.get(k2) or ...`` (the last value when none is truthy)."""
    value = None
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    return value


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _dict_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def parse_resume_state(payload: dict[str, Any]) -> ResumeState:
    """Parse a checkpoint dictionary into a structured ResumeState object."""
    phase, is_intra, iteration = _progress(
        payload, int(payload.get("iteration", payload.get("iter", 0)))
    )
    best_score = payload.get("best_score")
    raw_hist = payload.get("metric_history") or payload.get("history")
    history: list[dict[str, Any]] = (
        [dict(entry) for entry in raw_hist if isinstance(entry, dict)]
        if isinstance(raw_hist, list)
        else []
    )
    gov_state = _first(payload, "governor_state", "governor")

    return ResumeState(
        epoch=int(payload.get("epoch", 0)),
        model_state_dict=_model_state(payload),
        progress=ResumeProgress(iteration=iteration, is_intra_epoch=is_intra, phase=phase),
        optim=ResumeOptimStates(
            optimizer_state_dict=_dict_or_none(
                _first(payload, "optimizer_state_dict", "optimizer")
            ),
            scheduler_state_dict=_dict_or_none(
                _first(payload, "scheduler_state_dict", "scheduler")
            ),
            scaler_state_dict=_dict_or_none(_first(payload, "scaler_state_dict", "scaler")),
        ),
        governance=ResumeGovernance(
            governor_state=gov_state if isinstance(gov_state, dict) else {},
            sota_achieved=bool(payload.get("sota_achieved", False)),
            best_score=float(best_score) if isinstance(best_score, (int, float)) else None,
            metric_history=history,
        ),
        accumulators=ResumeAccumulators(
            train_metrics=_dict_copy(_first(payload, "train_metrics", "current_metrics")),
            train_accumulators=_dict_copy(payload.get("train_accumulators")),
            val_accumulators=_dict_copy(_first(payload, "val_accumulators", "val_metrics")),
        ),
    )


def _stretch_phases(phases_in: list, ratio: float, current_total_steps: int) -> list:
    """Scale each phase ``end_step`` by ``ratio``; the last phase ends on the final step.

    The phase dicts are updated in place; the returned list is a copy of ``phases_in``.
    """
    phases = list(phases_in)
    for i, phase in enumerate(phases):
        if isinstance(phase, dict) and "end_step" in phase:
            if i == len(phases) - 1:
                phase["end_step"] = current_total_steps - 1
            else:
                old_end = phase["end_step"]
                scaled_end = round((old_end + 1) * ratio - 1)
                phase["end_step"] = max(0, min(current_total_steps - 1, scaled_end))
    return phases


def stretch_scheduler_runway(
    scheduler: Any,
    state_dict: dict[str, Any],
    current_total_steps: int,
    expected_step: int | None = None,
) -> None:
    """Load scheduler state, stretching total steps and phases proportionally.

    The stretch applies only when the step count changed since the checkpoint was written.
    """
    if not isinstance(state_dict, dict):
        return

    sched_dict = dict(state_dict)

    if "total_steps" in sched_dict:
        old_total = sched_dict["total_steps"]
        old_last = sched_dict.get("last_epoch", 0)

        if old_total != current_total_steps and old_total > 0:
            ratio = current_total_steps / float(old_total)
            sched_dict["total_steps"] = current_total_steps

            if expected_step is not None:
                new_last = expected_step
            else:
                new_last = round(old_last * ratio)

            new_last = max(0, min(current_total_steps - 1, new_last))
            sched_dict["last_epoch"] = new_last
            sched_dict["_step_count"] = new_last + 1

            if "_schedule_phases" in sched_dict and isinstance(
                sched_dict["_schedule_phases"], list
            ):
                sched_dict["_schedule_phases"] = _stretch_phases(
                    sched_dict["_schedule_phases"], ratio, current_total_steps
                )

            logger.info(
                "Stretched scheduler runway: %d -> %d steps (last_epoch %d -> %d)",
                old_total,
                current_total_steps,
                old_last,
                new_last,
            )

        elif expected_step is not None:
            old_last = sched_dict.get("last_epoch", 0)
            if old_last != expected_step:
                logger.info(
                    "Re-anchoring de-synced scheduler step (%d vs expected %d)",
                    old_last,
                    expected_step,
                )
                expected_clamped = max(0, min(current_total_steps - 1, expected_step))
                sched_dict["last_epoch"] = expected_clamped
                sched_dict["_step_count"] = expected_clamped + 1

    state_dict.update(sched_dict)
    try:
        scheduler.load_state_dict(sched_dict)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, RuntimeError) as exc:
        logger.warning("Error loading scheduler state dict: %s", exc)


def scale_resume_progress(
    resume_iteration: int,
    source_loader_len: int,
    new_loader_len: int,
    is_progress_snapshot: bool = False,
) -> int:
    """Scale the resume iteration when batch size or dataset fraction changed between runs."""
    if resume_iteration <= 0 or source_loader_len <= 0 or new_loader_len <= 0:
        return 0

    raw_ratio = resume_iteration / float(source_loader_len)
    if is_progress_snapshot and raw_ratio >= 0.999:
        # Completed progress snapshot
        return new_loader_len

    scaled = int(min(0.999, raw_ratio) * new_loader_len)
    return max(0, min(new_loader_len, scaled))
