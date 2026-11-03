"""Shared per-batch helpers for the training and validation loops.

Out-of-memory detection and relief, rank-0 detection, the intra-epoch progress checkpoint
payload and the batch telemetry line are the same for both loops.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
import torch.distributed as dist

from training.training.context import TrainingContext


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
    """Build the intra-epoch progress checkpoint payload.

    ``extra`` holds the phase-specific entries (the accumulator snapshot, and for validation
    the last training metrics).
    """
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_name": ctx.model_name,
        "model_state": model.state_dict(),
        "optimizer_state": ctx.optimizer.state_dict(),
        "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
        "governor_state": ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None,
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
