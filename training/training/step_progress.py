"""Shared per-batch helpers for the training and validation loops.

Out-of-memory detection and relief, rank-0 detection, the intra-epoch progress checkpoint
payload and the batch telemetry line are the same for both loops.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from training.training.context import TrainingContext


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


def is_oom_error(err: BaseException) -> bool:
    """Return True when ``err`` is a CUDA out-of-memory failure."""
    return "out of memory" in str(err).lower() or isinstance(err, torch.cuda.OutOfMemoryError)


def release_cuda_cache() -> None:
    """Collect garbage and empty the CUDA cache when a GPU is present."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_main_process() -> bool:
    """Return True unless this is a non-zero rank of an initialised process group."""
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


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
    }
    payload.update(extra)
    return payload


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
