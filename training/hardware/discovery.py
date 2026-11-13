"""Device and accelerator discovery for LemGendary Model Training Suite."""

import logging
from dataclasses import dataclass

import torch

from training.utils.lazy_imports import load_torch_directml

logger = logging.getLogger("lemtrain.hardware.discovery")


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware device capabilities and memory profile."""

    device: torch.device
    device_type: str
    device_count: int
    device_names: list[str]
    total_vram_gb: float
    capability: tuple[int, int]
    is_cuda: bool

    @property
    def is_mps(self) -> bool:
        """Whether the device is Apple Silicon Metal (MPS)."""
        return self.device_type == "mps"

    @property
    def is_cpu(self) -> bool:
        """Whether the device is the host CPU."""
        return self.device_type == "cpu"


_CACHED_DEVICE_INFO: dict[str | None, DeviceInfo] = {}


def _forced_device_info(force_device: str) -> DeviceInfo:
    """Describe the explicitly requested device (CUDA details only when CUDA is available)."""
    dev = torch.device(force_device)
    dev_type = dev.type
    is_cuda = dev_type == "cuda" and torch.cuda.is_available()
    count = torch.cuda.device_count() if is_cuda else 1
    names = [torch.cuda.get_device_name(i) for i in range(count)] if is_cuda else [str(dev)]
    vram = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if is_cuda else 0.0
    cap = torch.cuda.get_device_capability(0) if is_cuda else (0, 0)
    info = DeviceInfo(
        device=dev,
        device_type=dev_type,
        device_count=count,
        device_names=names,
        total_vram_gb=vram,
        capability=cap,
        is_cuda=is_cuda,
    )
    return info


def _cuda_device_info() -> DeviceInfo | None:
    """CUDA GPUs, after a pre-flight kernel probe (raises on sm_ kernel mismatch)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Pre-flight CUDA sanity probe
        try:
            probe = torch.ones(1, device=device) + 1.0
            torch.cuda.synchronize()
            del probe
        except RuntimeError as exc:
            err_msg = str(exc)
            if (
                "no kernel image is available" in err_msg
                or "cudaErrorNoKernelImageForDevice" in err_msg
            ):
                logger.error(
                    "No CUDA kernel image available for %s (Compute Capability sm_%d%d). "
                    "Current PyTorch binary lacks kernels for this architecture.",
                    names[0],
                    cap[0],
                    cap[1],
                )
                raise RuntimeError(f"CUDA kernel image incompatibility on {names[0]}") from exc
            logger.warning("Pre-flight CUDA probe raised non-fatal exception: %s", exc)

        logger.info(
            "Discovered %d CUDA GPU(s): %s (Compute sm_%d%d) | Total VRAM: %.1f GB",
            count,
            ", ".join(names),
            cap[0],
            cap[1],
            vram,
        )
        info = DeviceInfo(
            device=device,
            device_type="cuda",
            device_count=count,
            device_names=names,
            total_vram_gb=vram,
            capability=cap,
            is_cuda=True,
        )
        return info
    return None


def _mps_device_info() -> DeviceInfo | None:
    """Apple Silicon (Metal)."""
    if (
        hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        logger.info("Discovered Apple Silicon Metal (MPS) acceleration.")
        info = DeviceInfo(
            device=torch.device("mps"),
            device_type="mps",
            device_count=1,
            device_names=["Apple Silicon Metal"],
            total_vram_gb=0.0,
            capability=(0, 0),
            is_cuda=False,
        )
        return info
    return None


def _xpu_device_info() -> DeviceInfo | None:
    """Intel ARC / XPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        count = torch.xpu.device_count()
        names = [torch.xpu.get_device_name(i) for i in range(count)]
        logger.info("Discovered %d Intel XPU accelerator(s): %s", count, ", ".join(names))
        info = DeviceInfo(
            device=torch.device("xpu"),
            device_type="xpu",
            device_count=count,
            device_names=names,
            total_vram_gb=0.0,
            capability=(0, 0),
            is_cuda=False,
        )
        return info
    return None


DIRECTML_ERRORS = (ImportError, OSError, RuntimeError, AttributeError)
"""torch-directml missing (ImportError), its DLL not loadable against this torch (OSError,
RuntimeError) or a build without the device API (AttributeError)."""


def dml_device_info() -> DeviceInfo | None:
    """Microsoft DirectML through ``torch_directml`` (torch device type ``privateuseone``).

    ``torch_directml.is_available()``, ``device_count()``, ``device_name(index)``,
    ``default_device()`` and ``device(index)`` (torch-directml 0.2.x); torch has no
    ``torch.dml`` attribute and no ``"dml"`` device type.
    """
    try:
        directml = load_torch_directml()
        if not directml.is_available():
            return None
        index = int(directml.default_device())
        count = int(directml.device_count())
        names = [str(directml.device_name(i)) for i in range(count)]
        device = directml.device(index)
    except DIRECTML_ERRORS as exc:
        logger.debug("DirectML (torch-directml) unavailable: %s", exc)
        return None
    name = names[index] if 0 <= index < count else "Microsoft DirectML"
    logger.info("Discovered Microsoft DirectML acceleration: %s (%s).", name, device)
    return DeviceInfo(
        device=torch.device(device),
        device_type="dml",
        device_count=count,
        device_names=names,
        total_vram_gb=0.0,
        capability=(0, 0),
        is_cuda=False,
    )


def _cpu_device_info() -> DeviceInfo:
    """CPU fallback."""
    logger.warning("No GPU/NPU accelerator discovered. Defaulting to CPU execution.")
    info = DeviceInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        device_count=1,
        device_names=["Host CPU"],
        total_vram_gb=0.0,
        capability=(0, 0),
        is_cuda=False,
    )
    return info


_ACCELERATOR_PROBES = (_cuda_device_info, _mps_device_info, _xpu_device_info, dml_device_info)


def _auto_device_info() -> DeviceInfo:
    """Return the first accelerator found (CUDA, MPS, XPU, DirectML), else the CPU."""
    for probe in _ACCELERATOR_PROBES:
        info = probe()
        if info is not None:
            return info
    return _cpu_device_info()


def discover_device(force_device: str | None = None) -> DeviceInfo:
    """
    Probe host hardware and select the optimal execution accelerator.

    Checks CUDA, Apple Silicon (MPS), Intel XPU, DirectML, and falls back to CPU.
    """
    if force_device in _CACHED_DEVICE_INFO:
        return _CACHED_DEVICE_INFO[force_device]
    if force_device:
        info = _forced_device_info(force_device)
    else:
        info = _auto_device_info()
    _CACHED_DEVICE_INFO[force_device] = info
    return info
