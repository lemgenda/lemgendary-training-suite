"""Training summary and execution result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSummary:
    """Summary record produced upon completion of training."""

    model_name: str
    final_epoch: int
    best_metrics: dict[str, float]
    total_time: float
    status: str
    stop_reason: str = ""
    """Why the epoch loop ended (see ``training.epoch_budget.StopReason``); empty when cancelled."""
