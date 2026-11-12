"""One device-memory adapter for every accelerator backend the suite trains on.

The VRAM audit (:func:`training.hardware.probe.audit_hardware_vram`), which sizes the
batch at start-up, after a resolution jump and on resume, and the out-of-memory relief
read device memory through :class:`DeviceMemory` instead of calling ``torch.cuda``
directly, so the same refit runs on every backend:

* ``cuda``: NVIDIA CUDA and AMD ROCm (HIP builds of torch expose the same ``torch.cuda``
  API). Free memory is ``torch.cuda.mem_get_info`` plus the allocator's unused reserved
  cache; peaks are tracked natively.
* ``xpu``: Intel XPU through ``torch.xpu.mem_get_info``; peaks are tracked natively.
* ``mps``: Apple Silicon. Unified memory is shared with the host, so free memory cannot be
  queried; the estimate is ``min(recommended_max_memory - current_allocated_memory, host
  available RAM)``. MPS has no peak statistic: the probe samples the allocation at its
  observation points (:meth:`DeviceMemory.observe`).
* ``dml``: DirectML (``torch-directml``, device type ``privateuseone``) reports no memory
  at all; the budget is the conservative ``hardware.directml_assumed_free_gb`` (default
  :data:`DIRECTML_ASSUMED_FREE_GB`) and the per-sample cost falls back to the probe's
  resolution-scaled formula.
* ``cpu``: no device budget; every operation is a no-op.

Every estimate (any budget that is not a direct free-memory reading, including a CUDA or
XPU query that failed) is logged when it is taken.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import psutil
import torch

logger = logging.getLogger("lemtrain.hardware.device_memory")

DIRECTML_ASSUMED_FREE_GB = 2.0
"""DirectML budget when ``hardware.directml_assumed_free_gb`` is unset: the smallest VRAM
of the DirectX 12 GPUs the suite targets, so the refit never overcommits an unknown card."""

_GIB = 1024**3

OOM_MESSAGE_MARKERS: tuple[str, ...] = (
    "out of memory",  # CUDA, ROCm, MPS ("MPS backend out of memory"), XPU
    "not enough gpu video memory",  # DirectML
)
"""Lower-case fragments of the out-of-memory errors of every backend."""


@dataclass(frozen=True)
class MemoryBudget:
    """Free and total device memory in bytes; ``measured`` is False for an estimate."""

    free_bytes: int
    total_bytes: int
    measured: bool

    @property
    def free_gb(self) -> float:
        """Free memory in GiB."""
        return self.free_bytes / _GIB

    @property
    def total_gb(self) -> float:
        """Total memory in GiB."""
        return self.total_bytes / _GIB


class DeviceMemory(Protocol):
    """Memory queries and controls of one device backend."""

    name: str

    def budget(self) -> MemoryBudget | None:
        """Free and total memory, or ``None`` when the device has no memory budget (CPU)."""
        ...

    def device_name(self) -> str:
        """Human-readable device name for the probe report."""
        ...

    def device_count(self) -> int:
        """Devices of this backend visible to the process."""
        ...

    def synchronize(self) -> None:
        """Wait for queued device work (surfaces asynchronous device faults)."""
        ...

    def release_cache(self) -> None:
        """Return the allocator's cached, unused blocks to the device."""
        ...

    def reset_peak(self) -> None:
        """Start a new peak-allocation window."""
        ...

    def observe(self) -> None:
        """Record the current allocation into the peak (backends without peak tracking)."""
        ...

    def allocated(self) -> int:
        """Bytes currently allocated by tensors."""
        ...

    def peak_allocated(self) -> int:
        """Highest allocation since :meth:`reset_peak` (0 when the backend reports none)."""
        ...


class CudaMemory(DeviceMemory):
    """NVIDIA CUDA and AMD ROCm devices through ``torch.cuda``."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.name = "rocm" if getattr(torch.version, "hip", None) else "cuda"

    def budget(self) -> MemoryBudget | None:
        unused_reserved = torch.cuda.memory_reserved(self.index) - torch.cuda.memory_allocated(
            self.index
        )
        try:
            free, total = torch.cuda.mem_get_info(self.index)
        except RuntimeError as mem_err:
            total = int(torch.cuda.get_device_properties(self.index).total_memory)
            reserved = int(torch.cuda.memory_reserved(self.index))
            logger.warning(
                "[VRAM] %s mem_get_info failed (%s); estimating free memory as total minus "
                "reserved (%.2f GB).",
                self.name,
                mem_err,
                (total - reserved) / _GIB,
            )
            return MemoryBudget(max(0, total - reserved), total, measured=False)
        return MemoryBudget(int(free) + max(0, int(unused_reserved)), int(total), measured=True)

    def device_name(self) -> str:
        return str(torch.cuda.get_device_name(self.index))

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.index)

    def release_cache(self) -> None:
        torch.cuda.empty_cache()

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.index)

    def observe(self) -> None:
        """Peaks are tracked by the CUDA caching allocator itself."""

    def allocated(self) -> int:
        return int(torch.cuda.memory_allocated(self.index))

    def peak_allocated(self) -> int:
        return int(torch.cuda.max_memory_allocated(self.index))


class XpuMemory(DeviceMemory):
    """Intel XPU devices through ``torch.xpu``."""

    name = "xpu"

    def __init__(self, index: int = 0) -> None:
        self.index = index

    @staticmethod
    def _xpu() -> Any:
        return torch.xpu

    def budget(self) -> MemoryBudget | None:
        xpu = self._xpu()
        try:
            free, total = xpu.mem_get_info(self.index)
        except RuntimeError as mem_err:
            total = int(xpu.get_device_properties(self.index).total_memory)
            reserved = int(xpu.memory_reserved(self.index))
            logger.warning(
                "[VRAM] xpu mem_get_info failed (%s); estimating free memory as total minus "
                "reserved (%.2f GB).",
                mem_err,
                (total - reserved) / _GIB,
            )
            return MemoryBudget(max(0, total - reserved), total, measured=False)
        unused_reserved = int(xpu.memory_reserved(self.index)) - int(
            xpu.memory_allocated(self.index)
        )
        return MemoryBudget(int(free) + max(0, unused_reserved), int(total), measured=True)

    def device_name(self) -> str:
        return str(self._xpu().get_device_name(self.index))

    def device_count(self) -> int:
        return int(self._xpu().device_count())

    def synchronize(self) -> None:
        self._xpu().synchronize(self.index)

    def release_cache(self) -> None:
        self._xpu().empty_cache()

    def reset_peak(self) -> None:
        self._xpu().reset_peak_memory_stats(self.index)

    def observe(self) -> None:
        """Peaks are tracked by the XPU caching allocator itself."""

    def allocated(self) -> int:
        return int(self._xpu().memory_allocated(self.index))

    def peak_allocated(self) -> int:
        return int(self._xpu().max_memory_allocated(self.index))


class MpsMemory(DeviceMemory):
    """Apple Silicon (Metal) unified memory; free memory is a logged estimate."""

    name = "mps"

    def __init__(self) -> None:
        self._peak = 0

    def budget(self) -> MemoryBudget | None:
        total = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.current_allocated_memory())
        host_available = int(psutil.virtual_memory().available)
        free = max(0, min(total - allocated, host_available))
        logger.info(
            "[VRAM] mps shares unified memory with the host; free memory is estimated as "
            "min(recommended working set - allocated, host available RAM) = %.2f GB.",
            free / _GIB,
        )
        return MemoryBudget(free, total, measured=False)

    def device_name(self) -> str:
        return "Apple Silicon Metal"

    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def release_cache(self) -> None:
        torch.mps.empty_cache()

    def reset_peak(self) -> None:
        self._peak = self.allocated()

    def observe(self) -> None:
        self._peak = max(self._peak, self.allocated())

    def allocated(self) -> int:
        return int(torch.mps.current_allocated_memory())

    def peak_allocated(self) -> int:
        return self._peak


class DirectMLMemory(DeviceMemory):
    """DirectML devices: no memory query exists, so the budget is a configured estimate."""

    name = "dml"

    def __init__(self, assumed_free_gb: float = DIRECTML_ASSUMED_FREE_GB) -> None:
        self.assumed_free_bytes = int(max(0.0, float(assumed_free_gb)) * _GIB)

    def budget(self) -> MemoryBudget | None:
        logger.info(
            "[VRAM] DirectML reports no device memory; assuming %.2f GB free "
            "(hardware.directml_assumed_free_gb).",
            self.assumed_free_bytes / _GIB,
        )
        return MemoryBudget(self.assumed_free_bytes, self.assumed_free_bytes, measured=False)

    def device_name(self) -> str:
        return "Microsoft DirectML"

    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        """DirectML exposes no device synchronization; work completes on readback."""

    def release_cache(self) -> None:
        """DirectML exposes no allocator cache control."""

    def reset_peak(self) -> None:
        """DirectML reports no allocation statistics."""

    def observe(self) -> None:
        """DirectML reports no allocation statistics."""

    def allocated(self) -> int:
        return 0

    def peak_allocated(self) -> int:
        return 0


class CpuMemory(DeviceMemory):
    """Host CPU execution: no device memory budget; every control is a no-op."""

    name = "cpu"

    def budget(self) -> MemoryBudget | None:
        return None

    def device_name(self) -> str:
        return "CPU"

    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        """Nothing is queued on the host."""

    def release_cache(self) -> None:
        """The host has no device allocator cache."""

    def reset_peak(self) -> None:
        """No device allocations are tracked on the host."""

    def observe(self) -> None:
        """No device allocations are tracked on the host."""

    def allocated(self) -> int:
        return 0

    def peak_allocated(self) -> int:
        return 0


def directml_assumed_free_gb(config: dict[str, Any] | None) -> float:
    """``hardware.directml_assumed_free_gb`` from ``config``, else the documented default."""
    hardware = (config or {}).get("hardware") or {}
    value = hardware.get("directml_assumed_free_gb") if isinstance(hardware, dict) else None
    return (
        float(value) if isinstance(value, (int, float)) and value > 0 else DIRECTML_ASSUMED_FREE_GB
    )


def device_memory_for(device: torch.device, config: dict[str, Any] | None = None) -> DeviceMemory:
    """The memory adapter of ``device`` (CPU no-op for unknown device types)."""
    index = device.index if device.index is not None else 0
    if device.type == "cuda" and torch.cuda.is_available():
        return CudaMemory(index)
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return XpuMemory(index)
    if device.type == "mps" and torch.backends.mps.is_available():
        return MpsMemory()
    if device.type in ("dml", "privateuseone"):
        return DirectMLMemory(directml_assumed_free_gb(config))
    return CpuMemory()


def present_device_memories() -> list[DeviceMemory]:
    """Adapters of every accelerator backend present in this process (cache control)."""
    present: list[DeviceMemory] = []
    if torch.cuda.is_available():
        present.append(CudaMemory())
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        present.append(XpuMemory())
    if torch.backends.mps.is_available():
        present.append(MpsMemory())
    return present


def release_device_memory(purpose: str) -> None:
    """Collect garbage and release every present backend's allocator cache (logged on error)."""
    gc.collect()
    for memory in present_device_memories():
        try:
            memory.release_cache()
        except RuntimeError as exc:
            logger.debug("%s cache release failed during %s: %s", memory.name, purpose, exc)


def is_out_of_memory(err: BaseException) -> bool:
    """Whether ``err`` is an out-of-memory failure of any device backend."""
    if isinstance(err, torch.OutOfMemoryError):
        return True
    message = str(err).lower()
    return any(marker in message for marker in OOM_MESSAGE_MARKERS)
