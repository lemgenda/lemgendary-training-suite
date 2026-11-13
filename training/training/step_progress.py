"""Shared per-batch helpers for the training and validation loops.

Rank-0 detection, the intra-epoch progress checkpoint
payload and the batch telemetry line are the same for both loops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from training.loss_emd import EMD_FORM_KEY
from training.run_settings.record import PAYLOAD_KEY as RUN_SETTINGS_KEY
from training.training.context import TrainingContext
from training.utils.rng_state import rng_payload


@dataclass
class BatchClock:
    """Batch count, resume offset and timing of one loop over a data loader."""

    num_batches: int
    start_time: float
    last_log_time: float
    resume_step: int = 0

    def skip_resumed(self, batch_idx: int, tag: str, phase: str) -> bool:
        """Return True for batches finished before an intra-epoch resume.

        Prints a fast-forward line every ``resume_step // 20`` batches (1 to 500) and on the
        last skipped batch.
        """
        if batch_idx >= self.resume_step:
            return False
        ff_interval = max(1, min(500, self.resume_step // 20))
        if (batch_idx + 1) % ff_interval == 0 or (batch_idx + 1) == self.resume_step:
            total = self.num_batches
            ff_pct = ((batch_idx + 1) / max(1, total)) * 100.0
            print(
                f"  [FAST-FORWARD] [{tag} {batch_idx + 1:>{len(str(total))}}/{total}] "
                f"({ff_pct:.1f}%) Skipping processed {phase} batches...",
                flush=True,
            )
        if (batch_idx + 1) == self.resume_step:
            # Fast-forward time must not count towards s/it and ETA of the live batches.
            self.start_time = time.time()
            self.last_log_time = self.start_time
        return True

    def rate(self, done: int, now: float) -> tuple[float, float]:
        """Return ``(iterations per second since start, elapsed seconds)`` after ``done`` steps.

        Only the steps run in this session (``done - resume_step``, at least 1) count.
        """
        elapsed = now - self.start_time
        return max(1, done - self.resume_step) / max(0.001, elapsed), elapsed


def resume_iteration(resume_state: Any, phase: str, handled: bool) -> int:
    """Return the batches already done in ``phase`` for an unhandled intra-epoch resume."""
    if resume_state is None or handled:
        return 0
    if resume_state.is_intra_epoch and resume_state.phase == phase and resume_state.iteration > 0:
        return resume_state.iteration
    return 0


def is_main_process() -> bool:
    """Return True unless this is a non-zero rank of an initialised process group."""
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


DDP_DECISION_EVERY = 25
"""Under DDP the ranks agree on a stop or save decision only every this many steps."""


def world_size() -> int:
    """Processes of the initialised process group (1 without one)."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def any_rank(flag: bool, device: torch.device) -> bool:
    """True when ``flag`` is true on any rank (a collective call under DDP)."""
    if world_size() <= 1:
        return flag
    agreed = torch.tensor([1 if flag else 0], device=device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MAX)
    return bool(agreed.item())


def decision_step(step: int) -> bool:
    """Whether ``step`` is a step the ranks agree on (every step without DDP)."""
    return world_size() <= 1 or step % DDP_DECISION_EVERY == 0


def deadline_reached(ctx: TrainingContext, step: int) -> bool:
    """Whether the Kaggle session deadline passed, agreed by all ranks at ``step``.

    Every rank calls this on every step; under DDP only decision steps make the collective
    call, so no rank stops alone.
    """
    deadline = ctx.state.session_deadline
    if deadline is None or not decision_step(step):
        return False
    return any_rank(deadline.expired(), ctx.hardware.device_info.device)


def progress_payload(
    ctx: TrainingContext,
    epoch: int,
    model: torch.nn.Module,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build a checkpoint payload (intra-epoch progress or end-of-epoch latest/best).

    ``extra`` holds the phase-specific entries (the accumulator snapshot, and for validation
    the last training metrics; the best/current metrics at the end of an epoch).
    """
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_name": ctx.model_name,
        "model_state": model.state_dict(),
        "optimizer_state": ctx.models.optimizer.state_dict(),
        "scheduler_state": ctx.models.scheduler.state_dict()
        if ctx.models.scheduler is not None
        else None,
        "governor_state": ctx.governance.governor.get_state()
        if hasattr(ctx.governance.governor, "get_state")
        else None,
        # AMP loss scale; without it a resumed fp16 run restarts at 65536 and skips steps.
        "scaler_state_dict": ctx.models.scaler.state_dict()
        if ctx.models.scaler is not None
        else None,
        # EMD form of the quality loss, fixed for the run (a resume keeps it).
        EMD_FORM_KEY: getattr(ctx.models.criterion, "emd_form", None),
        # Python, NumPy, torch CPU and device generators of every rank (collective under
        # DDP); a resume continues from them.
        **rng_payload(),
    }
    tracker = ctx.state.run_settings
    if tracker is not None:
        # Frozen settings (+ hash) and the epoch settings this epoch trains with; a resume
        # finishes the epoch on them and switches to the configured ones next epoch.
        payload[RUN_SETTINGS_KEY] = tracker.record(epoch).to_payload()
    payload.update(extra)
    return payload


def progress_save_agreed(due: bool, step: int, device: torch.device) -> bool:
    """Whether all ranks save intra-epoch progress at ``step`` (``due``: this rank's guardrails).

    The payload gathers every rank's generator state, so all ranks must build it together:
    under DDP the ranks agree only on decision steps (one collective call there).
    """
    if not decision_step(step):
        return False
    return any_rank(due, device)


def format_rate(rate: float) -> str:
    """Format a throughput as iterations per second, or seconds per iteration below 1 it/s."""
    return f"{rate:.2f} it/s" if rate >= 1.0 else f"{1.0 / max(1e-4, rate):.1f} s/it"


def format_eta(remaining: int, rate: float) -> str:
    """Format the time left for ``remaining`` iterations at ``rate`` it/s."""
    eta_s = int(remaining / max(1e-4, rate))
    eta_m, eta_sec = divmod(eta_s, 60)
    eta_h, eta_m = divmod(eta_m, 60)
    return f"ETA {eta_h}h{eta_m:02d}m" if eta_h > 0 else f"ETA {eta_m:02d}m{eta_sec:02d}s"


def batch_progress_line(
    tag: str,
    position: tuple[int, int],
    metrics: str,
    rate: float,
    elapsed: float,
) -> str:
    """Return the padded ``[TAG step/total]`` telemetry line.

    ``position`` is ``(step, total)``; without a known total the line omits the percentage
    and ETA. ``metrics`` is the phase-specific ``Loss: ... | ...`` segment.
    """
    step, total = position
    rate_str = format_rate(rate)
    if total <= 0:
        line = f"\r  [{tag} {step}] {metrics} | {rate_str} | {elapsed:.1f}s"
        return line.ljust(90)
    pct = (step / total) * 100.0
    eta_str = format_eta(max(0, total - step), rate)
    line = (
        f"\r  [{tag} {step:>{len(str(total))}}/{total}] ({pct:.2f}%) "
        f"{metrics} | {rate_str} | {eta_str} | {elapsed:.1f}s"
    )
    return line.ljust(110)
