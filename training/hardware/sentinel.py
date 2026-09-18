"""Proactive VRAM SentinelGuard and OOM prevention for LemGendary Model Training Suite."""

import gc
import logging
import torch

logger = logging.getLogger("lemtrain.hardware.sentinel")


class SentinelGuard:
    """
    Proactive memory sentinel monitoring GPU memory headroom before forward/backward steps,
    triggering emergency cache purges, batch throttling, and spatial vetoes before OOM crashes.
    """

    def __init__(self, device: torch.device, memory_ceiling_ratio: float = 0.90) -> None:
        self.device = device
        self.memory_ceiling_ratio = memory_ceiling_ratio
        self.is_cuda = device.type == "cuda" and torch.cuda.is_available()

    def check_memory(self) -> dict[str, float]:
        """Return allocated, reserved, and free memory metrics in gigabytes."""
        if not self.is_cuda:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0,
                "total_gb": 0.0,
                "usage_ratio": 0.0,
            }

        total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        free = total - reserved

        usage_ratio = reserved / total if total > 0 else 0.0

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "total_gb": total,
            "usage_ratio": usage_ratio,
        }

    def is_memory_critical(self) -> bool:
        """Check if memory usage exceeds configured ceiling ratio."""
        if not self.is_cuda:
            return False
        metrics = self.check_memory()
        return metrics["usage_ratio"] >= self.memory_ceiling_ratio

    def recover_memory(self) -> None:
        """Execute garbage collection and empty CUDA caching allocator."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
        logger.debug("SentinelGuard triggered cache purge and memory recovery.")

    def throttle_batch(self, current_batch: int, min_batch: int = 1) -> int:
        """Reduce batch size by half when memory headroom is compromised."""
        if current_batch <= min_batch:
            return min_batch
        new_batch = max(min_batch, current_batch // 2)
        logger.warning(
            "SentinelGuard throttled batch size from %d to %d due to memory pressure.",
            current_batch, new_batch
        )
        return new_batch

    def veto_resolution_jump(self, requested_res: int, current_res: int) -> tuple[bool, str]:
        """
        Evaluate whether a curriculum resolution jump should be vetoed based on free VRAM headroom.
        
        Returns:
            (is_vetoed, reason_string)
        """
        if not self.is_cuda:
            return False, "Non-CUDA device; spatial jump permitted."

        metrics = self.check_memory()
        if metrics["free_gb"] < 1.0 or metrics["usage_ratio"] >= self.memory_ceiling_ratio:
            reason = (
                f"VRAM headroom too tight ({metrics['free_gb']:.2f} GB free, "
                f"{metrics['usage_ratio']*100:.1f}% reserved). "
                f"Resolution jump from {current_res}px to {requested_res}px vetoed."
            )
            logger.warning("SentinelGuard vetoed spatial jump: %s", reason)
            return True, reason

        return False, "Memory headroom acceptable for spatial transition."
