"""Loss-spike and gradient norm anomaly detector with automated recovery.

Monitors batch-level loss progression and gradient norms during training.
Detects sudden divergence (loss > threshold multiplier of rolling median, or NaN/Inf)
and coordinates with lifecycle managers or rollback hooks to stabilize training.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import torch

logger = logging.getLogger("lemtrain.training.spike_detector")


@dataclass(frozen=True)
class AnomalyVerdict:
    """Outcome of evaluating a training step for anomalies."""

    is_spike: bool
    is_non_finite: bool
    reason: str | None = None
    rolling_median: float = 0.0
    current_value: float = 0.0


@dataclass
class SpikeThresholds:
    """Configurable sensitivity thresholds for loss and gradient norm spikes."""

    window_size: int = 50
    spike_multiplier: float = 3.5
    grad_norm_multiplier: float = 5.0
    min_history: int = 10


class LossSpikeDetector:
    """Tracks moving loss and gradient norms to detect malignant spikes."""

    def __init__(
        self,
        thresholds: SpikeThresholds | None = None,
    ) -> None:
        self.cfg = thresholds or SpikeThresholds()
        self._loss_history: deque[float] = deque(maxlen=self.cfg.window_size)
        self._grad_history: deque[float] = deque(maxlen=self.cfg.window_size)
        self.consecutive_spikes: int = 0
        self.total_spikes_detected: int = 0

    def compute_median_loss(self) -> float:
        """Return the median of tracked loss values."""
        if not self._loss_history:
            return 0.0
        sorted_losses = sorted(self._loss_history)
        n = len(sorted_losses)
        mid = n // 2
        if n % 2 == 1:
            return sorted_losses[mid]
        return 0.5 * (sorted_losses[mid - 1] + sorted_losses[mid])

    def compute_mean_grad_norm(self) -> float:
        """Return the mean of tracked gradient norms."""
        if not self._grad_history:
            return 0.0
        return sum(self._grad_history) / len(self._grad_history)

    def audit_step(
        self,
        loss_val: float | torch.Tensor,
        grad_norm: float | None = None,
    ) -> AnomalyVerdict:
        """Inspect current step loss and gradient norm for malignant anomalies."""
        val = float(loss_val.item()) if isinstance(loss_val, torch.Tensor) else float(loss_val)

        # Check finite bounds
        if not math.isfinite(val):
            self.consecutive_spikes += 1
            self.total_spikes_detected += 1
            logger.error("[SPIKE DETECTOR] Non-finite loss encountered: %s", val)
            return AnomalyVerdict(
                is_spike=True,
                is_non_finite=True,
                reason="Loss value is NaN or Inf",
                current_value=val,
            )

        if grad_norm is not None and not math.isfinite(grad_norm):
            self.consecutive_spikes += 1
            self.total_spikes_detected += 1
            logger.error("[SPIKE DETECTOR] Non-finite gradient norm encountered: %s", grad_norm)
            return AnomalyVerdict(
                is_spike=True,
                is_non_finite=True,
                reason="Gradient norm is NaN or Inf",
                current_value=grad_norm,
            )

        median_loss = self.compute_median_loss()
        mean_grad = self.compute_mean_grad_norm()

        # Check for sudden loss explosion
        if (
            len(self._loss_history) >= self.cfg.min_history
            and median_loss > 1e-6
            and val > (median_loss * self.cfg.spike_multiplier)
        ):
            self.consecutive_spikes += 1
            self.total_spikes_detected += 1
            logger.warning(
                "[SPIKE DETECTOR] Loss spike detected: %.4f vs median %.4f (factor %.2fx)",
                val,
                median_loss,
                val / median_loss,
            )
            return AnomalyVerdict(
                is_spike=True,
                is_non_finite=False,
                reason=(
                    f"Loss {val:.4f} exceeded {self.cfg.spike_multiplier}x "
                    f"median ({median_loss:.4f})"
                ),
                rolling_median=median_loss,
                current_value=val,
            )

        # Check for abnormal gradient norm jump
        if (
            grad_norm is not None
            and len(self._grad_history) >= self.cfg.min_history
            and mean_grad > 1e-6
            and grad_norm > (mean_grad * self.cfg.grad_norm_multiplier)
        ):
            self.consecutive_spikes += 1
            self.total_spikes_detected += 1
            logger.warning(
                "[SPIKE DETECTOR] Gradient norm spike detected: %.4f vs mean %.4f (factor %.2fx)",
                grad_norm,
                mean_grad,
                grad_norm / mean_grad,
            )
            return AnomalyVerdict(
                is_spike=True,
                is_non_finite=False,
                reason=(
                    f"Gradient norm {grad_norm:.4f} exceeded {self.cfg.grad_norm_multiplier}x "
                    f"mean ({mean_grad:.4f})"
                ),
                rolling_median=mean_grad,
                current_value=grad_norm,
            )

        # Healthy step: record history
        self._loss_history.append(val)
        if grad_norm is not None and grad_norm > 0:
            self._grad_history.append(grad_norm)
        self.consecutive_spikes = 0

        return AnomalyVerdict(
            is_spike=False,
            is_non_finite=False,
            rolling_median=median_loss,
            current_value=val,
        )

    def reset(self) -> None:
        """Clear anomaly history buffers."""
        self._loss_history.clear()
        self._grad_history.clear()
        self.consecutive_spikes = 0
