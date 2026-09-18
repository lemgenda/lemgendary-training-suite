"""Typed resume state, scheduler runway stretching, and iteration progress scaling."""

from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger("lemtrain.checkpoint.resume")


@dataclass
class ResumeState:
    """Typed container encapsulating all recoverable state from a training checkpoint."""

    epoch: int
    iteration: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None
    governor_state: dict[str, Any] = field(default_factory=dict)
    sota_achieved: bool = False
    best_score: float | None = None
    metric_history: list[dict[str, Any]] = field(default_factory=list)


def parse_resume_state(payload: dict[str, Any]) -> ResumeState:
    """Parse a checkpoint dictionary into a structured ResumeState object."""
    # 1. Resolve model state dict
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        model_sd = payload["model_state_dict"]
    elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
        model_sd = payload["state_dict"]
    elif "model" in payload and isinstance(payload["model"], dict):
        model_sd = payload["model"]
    else:
        # Assume whole dict is raw weights
        model_sd = payload

    epoch = int(payload.get("epoch", 0))
    iteration = int(payload.get("iteration", payload.get("iter", 0)))
    opt_sd = payload.get("optimizer_state_dict") or payload.get("optimizer")
    sched_sd = payload.get("scheduler_state_dict") or payload.get("scheduler")
    scaler_sd = payload.get("scaler_state_dict") or payload.get("scaler")
    gov_state = payload.get("governor_state") or payload.get("governor") or {}
    sota_achieved = bool(payload.get("sota_achieved", False))
    best_score = payload.get("best_score")

    history: list[dict[str, Any]] = []
    raw_hist = payload.get("metric_history") or payload.get("history")
    if isinstance(raw_hist, list):
        history = [item for item in raw_hist if isinstance(item, dict)]

    return ResumeState(
        epoch=epoch,
        iteration=iteration,
        model_state_dict=model_sd,
        optimizer_state_dict=opt_sd if isinstance(opt_sd, dict) else None,
        scheduler_state_dict=sched_sd if isinstance(sched_sd, dict) else None,
        scaler_state_dict=scaler_sd if isinstance(scaler_sd, dict) else None,
        governor_state=gov_state if isinstance(gov_state, dict) else {},
        sota_achieved=sota_achieved,
        best_score=float(best_score) if isinstance(best_score, (int, float)) else None,
        metric_history=history,
    )


def stretch_scheduler_runway(
    scheduler: Any,
    state_dict: dict[str, Any],
    current_total_steps: int,
    expected_step: int | None = None,
) -> None:
    """Load scheduler state dict, proportionally stretching total steps and scheduled phases if step counts shifted."""
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
                new_last = int(round(old_last * ratio))

            new_last = max(0, min(current_total_steps - 1, new_last))
            sched_dict["last_epoch"] = new_last
            sched_dict["_step_count"] = new_last + 1

            if "_schedule_phases" in sched_dict and isinstance(sched_dict["_schedule_phases"], list):
                phases = list(sched_dict["_schedule_phases"])
                for i, phase in enumerate(phases):
                    if isinstance(phase, dict) and "end_step" in phase:
                        if i == len(phases) - 1:
                            phase["end_step"] = current_total_steps - 1
                        else:
                            old_end = phase["end_step"]
                            scaled_end = int(round((old_end + 1) * ratio - 1))
                            phase["end_step"] = max(0, min(current_total_steps - 1, scaled_end))
                sched_dict["_schedule_phases"] = phases

            logger.info("Stretched scheduler runway: %d -> %d steps (last_epoch %d -> %d)", old_total, current_total_steps, old_last, new_last)

        elif expected_step is not None:
            old_last = sched_dict.get("last_epoch", 0)
            if old_last != expected_step:
                logger.info("Re-anchoring de-synced scheduler step (%d vs expected %d)", old_last, expected_step)
                expected_clamped = max(0, min(current_total_steps - 1, expected_step))
                sched_dict["last_epoch"] = expected_clamped
                sched_dict["_step_count"] = expected_clamped + 1

    state_dict.update(sched_dict)
    try:
        scheduler.load_state_dict(sched_dict)
    except Exception as exc:
        logger.warning("Error loading scheduler state dict: %s", exc)


def scale_resume_progress(
    resume_iteration: int,
    source_loader_len: int,
    new_loader_len: int,
    is_progress_snapshot: bool = False,
) -> int:
    """Scale resume iteration proportionally when batch size or dataset fraction shifts between runs."""
    if resume_iteration <= 0 or source_loader_len <= 0 or new_loader_len <= 0:
        return 0

    raw_ratio = resume_iteration / float(source_loader_len)
    if is_progress_snapshot and raw_ratio >= 0.999:
        # Completed progress snapshot
        return new_loader_len

    scaled = int(min(0.999, raw_ratio) * new_loader_len)
    return max(0, min(new_loader_len, scaled))
