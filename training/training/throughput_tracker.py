"""Throughput and Model FLOPs Utilization (MFU) instrumentation for LLM training.

Tracks tokens per second, step latency, and hardware compute utilization
derived from the PaLM floating-point operations formulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Theoretical peak dense TFLOPS per accelerator architecture (bfloat16 / fp16)
_PEAK_TFLOPS_MAP: dict[str, float] = {
    "gtx_1650": 6.0,  # NVIDIA GeForce GTX 1650 (FP32 baseline, no tensor cores)
    "t4": 65.0,  # NVIDIA Tesla T4 (FP16 Tensor Core)
    "a100": 312.0,  # NVIDIA A100 SXM4 (BF16/FP16 Tensor Core)
    "h100": 989.0,  # NVIDIA H100 SXM (BF16/FP16 Tensor Core without sparsity)
    "tpu_v5e": 197.0,  # Google TPU v5e (BF16 TensorCore per chip)
    "apple_silicon": 10.0,  # Apple M-series standard baseline
}


@dataclass(frozen=True)
class ThroughputReading:
    """Telemetry metrics captured across training iterations."""

    tokens_per_sec: float
    samples_per_sec: float
    mfu_percent: float
    elapsed_sec: float
    total_tokens: int


@dataclass(frozen=True)
class ModelArchConfig:
    """Transformer architecture dimensional parameters for FLOP calculations."""

    num_params: int = 1_500_000_000
    num_layers: int = 28
    num_heads: int = 12
    head_dim: int = 128


class ThroughputTracker:
    """Instruments token throughput and calculates Model FLOPs Utilization (MFU)."""

    def __init__(
        self,
        arch: ModelArchConfig | None = None,
        device_type: str = "gtx_1650",
        num_devices: int = 1,
    ) -> None:
        self.arch = arch or ModelArchConfig()
        self.num_devices = max(1, num_devices)

        dev_clean = device_type.lower().replace("-", "_").replace(" ", "_")
        self.peak_tflops = _PEAK_TFLOPS_MAP.get(dev_clean, 10.0)

        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._total_tokens = 0
        self._total_samples = 0

    def compute_flops_per_token(self, seq_len: int) -> float:
        """Compute theoretical minimum FLOPs required per token (forward + backward).

        Based on the PaLM formulation:
            FLOPs/token = 6 * N + 12 * L * h * q * t
        """
        dense_flops = 6.0 * float(self.arch.num_params)
        attn_flops = 12.0 * float(
            self.arch.num_layers * self.arch.num_heads * self.arch.head_dim * seq_len
        )
        return dense_flops + attn_flops

    def record_step(
        self,
        tokens_in_step: int,
        samples_in_step: int = 1,
        seq_len: int = 512,
    ) -> ThroughputReading:
        """Update tracker with the completed step and return throughput metrics."""
        now = time.perf_counter()
        dt = max(1e-5, now - self._last_time)
        self._last_time = now

        self._total_tokens += tokens_in_step
        self._total_samples += samples_in_step

        tokens_per_sec = float(tokens_in_step) / dt
        samples_per_sec = float(samples_in_step) / dt

        flops_per_tok = self.compute_flops_per_token(seq_len)
        achieved_flops = tokens_per_sec * flops_per_tok
        peak_hardware_flops = self.peak_tflops * 1e12 * float(self.num_devices)

        mfu = (achieved_flops / max(1.0, peak_hardware_flops)) * 100.0

        return ThroughputReading(
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            mfu_percent=mfu,
            elapsed_sec=dt,
            total_tokens=self._total_tokens,
        )

    def summary(self) -> dict[str, float]:
        """Return cumulative throughput statistics."""
        total_time = max(1e-5, time.perf_counter() - self._start_time)
        avg_tokens_sec = float(self._total_tokens) / total_time
        return {
            "total_tokens": float(self._total_tokens),
            "total_samples": float(self._total_samples),
            "avg_tokens_per_sec": avg_tokens_sec,
            "total_time_sec": total_time,
        }
