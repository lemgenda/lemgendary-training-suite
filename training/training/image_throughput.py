"""Throughput instrumentation for visual model training pipelines.

Instruments images per second, step latency, batch timing variance,
and GPU memory utilization to diagnose prefetch, decode, and pipeline bottlenecks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisionThroughputReading:
    """Telemetry reading capturing vision pipeline throughput metrics."""

    images_per_sec: float
    batches_per_sec: float
    step_latency_sec: float
    total_images: int
    gpu_mem_used_mb: float
    gpu_mem_max_mb: float


class ImageThroughputTracker:
    """Tracks images/sec, decode latency, and GPU memory metrics."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._total_images = 0
        self._total_batches = 0

    def reset(self) -> None:
        """Reset internal accumulator clocks."""
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._total_images = 0
        self._total_batches = 0

    def record_step(self, num_images: int) -> VisionThroughputReading:
        """Record completed step and compute instantaneous throughput."""
        now = time.perf_counter()
        dt = max(1e-5, now - self._last_time)
        self._last_time = now

        num_images = max(0, int(num_images))
        self._total_images += num_images
        self._total_batches += 1

        images_per_sec = float(num_images) / dt
        batches_per_sec = 1.0 / dt

        gpu_mem_used = 0.0
        gpu_mem_max = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
            gpu_mem_max = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)

        return VisionThroughputReading(
            images_per_sec=images_per_sec,
            batches_per_sec=batches_per_sec,
            step_latency_sec=dt,
            total_images=self._total_images,
            gpu_mem_used_mb=round(gpu_mem_used, 2),
            gpu_mem_max_mb=round(gpu_mem_max, 2),
        )

    def summary(self) -> dict[str, float]:
        """Compute aggregate average throughput metrics across entire session."""
        now = time.perf_counter()
        total_time = max(1e-5, now - self._start_time)
        return {
            "avg_images_per_sec": float(self._total_images) / total_time,
            "avg_batches_per_sec": float(self._total_batches) / total_time,
            "total_images": float(self._total_images),
            "total_elapsed_sec": total_time,
        }
