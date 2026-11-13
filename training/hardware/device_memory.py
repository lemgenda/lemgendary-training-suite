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
  itself. The budget is read on every call through a chain: an explicitly set
  ``hardware.directml_assumed_free_gb`` wins; else the DXGI per-process budget of the
  adapter (:mod:`training.hardware.dxgi_memory`, Windows): ``budget - current usage``,
  capped by host available RAM on an integrated (unified-memory) GPU, measured; else
  :data:`DIRECTML_DEDICATED_FRACTION` of the adapter's dedicated video memory (estimate);
  else :data:`DIRECTML_ASSUMED_FREE_GB`. The per-sample cost falls back to the probe's
  resolution-scaled formula (no allocation statistics).
* ``cpu``: no device budget; every operation is a no-op.

Every estimate (any budget that is not a direct free-memory reading, including a CUDA or
XPU query that failed) is logged when it is taken.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import psutil
import torch

from training.hardware.dxgi_memory import (
    AdapterMemory,
    AdapterReader,
    candidate_adapters,
    default_adapter_reader,
    is_unified_memory,
)
from training.utils.lazy_imports import load_torch_directml

logger = logging.getLogger("lemtrain.hardware.device_memory")

DIRECTML_ASSUMED_FREE_GB = 2.0
"""DirectML budget when neither DXGI nor the adapter description answers and
``hardware.directml_assumed_free_gb`` is unset: the smallest VRAM of the DirectX 12 GPUs
the suite targets, so the refit never overcommits an unknown card."""

DIRECTML_DEDICATED_FRACTION = 0.8
"""Share of the adapter's dedicated video memory assumed free when DXGI has no budget."""

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


def directml_device_name(index: int) -> str | None:
    """``torch_directml.device_name(index)``, or ``None`` without torch-directml (logged)."""
    try:
        return str(load_torch_directml().device_name(index))
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("[VRAM] torch-directml device name unavailable: %s", exc)
        return None


class DirectMLMemory(DeviceMemory):
    """DirectML devices: free memory from the DXGI budget, else a logged estimate.

    ``override_free_gb`` (an explicitly set ``hardware.directml_assumed_free_gb``) wins over
    every reading. ``read_adapters`` reads the adapters' video memory (``None`` skips DXGI,
    as on every platform but Windows); ``name_of`` maps the device index to its torch-directml
    name, which selects the DXGI adapter.
    """

    name = "dml"

    def __init__(
        self,
        index: int = 0,
        *,
        override_free_gb: float | None = None,
        read_adapters: AdapterReader | None = None,
        name_of: Callable[[int], str | None] = directml_device_name,
    ) -> None:
        self.index = index
        self.override_free_bytes = (
            int(override_free_gb * _GIB) if override_free_gb is not None else None
        )
        self._read_adapters = read_adapters
        self._name_of = name_of

    def budget(self) -> MemoryBudget | None:
        if self.override_free_bytes is not None:
            logger.info(
                "[VRAM] DirectML: hardware.directml_assumed_free_gb is set; assuming %.2f GB free.",
                self.override_free_bytes / _GIB,
            )
            return MemoryBudget(self.override_free_bytes, self.override_free_bytes, False)
        candidates = self._candidates()
        return (
            self._dxgi_budget(candidates)
            or self._dedicated_estimate(candidates)
            or self._assumed_default()
        )

    def _candidates(self) -> list[AdapterMemory]:
        """The DXGI adapters that may be this device (empty when DXGI is unavailable)."""
        if self._read_adapters is None:
            return []
        try:
            adapters = self._read_adapters()
        except (OSError, AttributeError, ValueError) as exc:
            logger.warning("[VRAM] DirectML: DXGI query failed (%s); estimating.", exc)
            return []
        return candidate_adapters(adapters, self._name_of(self.index))

    @staticmethod
    def _dxgi_budget(candidates: list[AdapterMemory]) -> MemoryBudget | None:
        """``budget - usage`` of the candidate with the smallest DXGI budget (measured)."""
        budgeted = [
            (adapter, adapter.budget_bytes)
            for adapter in candidates
            if adapter.budget_bytes is not None
        ]
        if not budgeted:
            return None
        adapter, total = min(budgeted, key=lambda item: item[1])
        usage = adapter.usage_bytes or 0
        free = max(0, total - usage)
        unified = is_unified_memory(adapter)
        if unified:
            free = min(free, int(psutil.virtual_memory().available))
        logger.info(
            "[VRAM] DirectML %s: DXGI budget %.2f GB, in use %.2f GB, free %.2f GB%s.",
            adapter.name,
            total / _GIB,
            usage / _GIB,
            free / _GIB,
            " (integrated GPU, capped by host available RAM)" if unified else "",
        )
        return MemoryBudget(free, total, measured=True)

    @staticmethod
    def _dedicated_estimate(candidates: list[AdapterMemory]) -> MemoryBudget | None:
        """:data:`DIRECTML_DEDICATED_FRACTION` of the smallest dedicated memory (estimate)."""
        sized = [adapter for adapter in candidates if adapter.dedicated_bytes > 0]
        if not sized:
            return None
        adapter = min(sized, key=lambda item: item.dedicated_bytes)
        free = int(adapter.dedicated_bytes * DIRECTML_DEDICATED_FRACTION)
        logger.info(
            "[VRAM] DirectML %s: no DXGI budget; assuming %.2f GB free (%.0f%% of %.2f GB "
            "dedicated video memory).",
            adapter.name,
            free / _GIB,
            DIRECTML_DEDICATED_FRACTION * 100,
            adapter.dedicated_bytes / _GIB,
        )
        return MemoryBudget(free, adapter.dedicated_bytes, measured=False)

    @staticmethod
    def _assumed_default() -> MemoryBudget:
        """The documented :data:`DIRECTML_ASSUMED_FREE_GB` (estimate)."""
        free = int(DIRECTML_ASSUMED_FREE_GB * _GIB)
        logger.info(
            "[VRAM] DirectML reports no device memory and DXGI did not answer; assuming "
            "%.2f GB free (default; hardware.directml_assumed_free_gb overrides).",
            free / _GIB,
        )
        return MemoryBudget(free, free, measured=False)

    def device_name(self) -> str:
        return self._name_of(self.index) or "Microsoft DirectML"

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


def directml_free_gb_override(config: dict[str, Any] | None) -> float | None:
    """An explicitly set positive ``hardware.directml_assumed_free_gb``, else ``None``."""
    hardware = (config or {}).get("hardware") or {}
    value = hardware.get("directml_assumed_free_gb") if isinstance(hardware, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


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
        return DirectMLMemory(
            index,
            override_free_gb=directml_free_gb_override(config),
            read_adapters=default_adapter_reader(),
        )
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
