"""Device and accelerator discovery for LemGendary Model Training Suite."""

from dataclasses import dataclass
import logging
import sys
import torch
from typing import Any

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
    is_mps: bool
    is_cpu: bool


def discover_device(force_device: str | None = None) -> DeviceInfo:
    """
    Probe host hardware and select the optimal execution accelerator.
    
    Checks CUDA, Apple Silicon (MPS), Intel XPU, DirectML, and falls back to CPU.
    """
    if force_device:
        dev = torch.device(force_device)
        dev_type = dev.type
        is_cuda = dev_type == "cuda" and torch.cuda.is_available()
        count = torch.cuda.device_count() if is_cuda else 1
        names = [torch.cuda.get_device_name(i) for i in range(count)] if is_cuda else [str(dev)]
        vram = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if is_cuda else 0.0
        cap = torch.cuda.get_device_capability(0) if is_cuda else (0, 0)
        return DeviceInfo(
            device=dev,
            device_type=dev_type,
            device_count=count,
            device_names=names,
            total_vram_gb=vram,
            capability=cap,
            is_cuda=is_cuda,
            is_mps=dev_type == "mps",
            is_cpu=dev_type == "cpu",
        )

    # 1. CUDA Accelerators
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
        except Exception as exc:
            err_msg = str(exc)
            if "no kernel image is available" in err_msg or "cudaErrorNoKernelImageForDevice" in err_msg:
                logger.error(
                    "No CUDA kernel image available for %s (Compute Capability sm_%d%d). "
                    "Current PyTorch binary lacks kernels for this architecture.",
                    names[0], cap[0], cap[1]
                )
                raise RuntimeError(f"CUDA kernel image incompatibility on {names[0]}") from exc
            logger.warning("Pre-flight CUDA probe raised non-fatal exception: %s", exc)

        logger.info(
            "Discovered %d CUDA GPU(s): %s (Compute sm_%d%d) | Total VRAM: %.1f GB",
            count, ", ".join(names), cap[0], cap[1], vram
        )
        return DeviceInfo(
            device=device,
            device_type="cuda",
            device_count=count,
            device_names=names,
            total_vram_gb=vram,
            capability=cap,
            is_cuda=True,
            is_mps=False,
            is_cpu=False,
        )

    # 2. Apple Silicon (Metal)
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Discovered Apple Silicon Metal (MPS) acceleration.")
        return DeviceInfo(
            device=torch.device("mps"),
            device_type="mps",
            device_count=1,
            device_names=["Apple Silicon Metal"],
            total_vram_gb=0.0,
            capability=(0, 0),
            is_cuda=False,
            is_mps=True,
            is_cpu=False,
        )

    # 3. Intel ARC / XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        count = torch.xpu.device_count()
        names = [torch.xpu.get_device_name(i) for i in range(count)]
        logger.info("Discovered %d Intel XPU accelerator(s): %s", count, ", ".join(names))
        return DeviceInfo(
            device=torch.device("xpu"),
            device_type="xpu",
            device_count=count,
            device_names=names,
            total_vram_gb=0.0,
            capability=(0, 0),
            is_cuda=False,
            is_mps=False,
            is_cpu=False,
        )

    # 4. Microsoft DirectML
    if hasattr(torch, "dml") and getattr(torch, "dml").is_available():
        logger.info("Discovered Microsoft DirectML acceleration.")
        return DeviceInfo(
            device=torch.device("dml"),
            device_type="dml",
            device_count=1,
            device_names=["Microsoft DirectML"],
            total_vram_gb=0.0,
            capability=(0, 0),
            is_cuda=False,
            is_mps=False,
            is_cpu=False,
        )

    # 5. CPU Fallback
    logger.warning("No GPU/NPU accelerator discovered. Defaulting to CPU execution.")
    return DeviceInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        device_count=1,
        device_names=["Host CPU"],
        total_vram_gb=0.0,
        capability=(0, 0),
        is_cuda=False,
        is_mps=False,
        is_cpu=True,
    )
