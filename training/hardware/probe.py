"""Hardware VRAM probing and dynamic batch-size calculation for LemGendary Model Training Suite."""

import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, NamedTuple

import psutil
import torch
from torch import nn

from training.hardware.device_memory import DeviceMemory, device_memory_for

logger = logging.getLogger("lemtrain.hardware.probe")

# Failure modes of an empirical forward/backward VRAM probe: CUDA/OOM/kernel errors
# (torch.cuda.OutOfMemoryError and AcceleratorError subclass RuntimeError), and model/input
# contract mismatches (shape/dtype/arity/output-structure errors).
_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    IndexError,
    KeyError,
)

_GIB = 1024**3

# Model-key fragments of the heavy restoration / face architectures.
_RESTORATION_KEYS = (
    "nafnet",
    "mprnet",
    "mirnet",
    "ffanet",
    "codeformer",
    "film_restorer",
    "parsenet",
    "ultrazoom",
    "multitask",
    "retinaface",
)

# Forex per-device batch tiers: (batch, minimum total VRAM GB); the dynamic estimate must
# also reach the batch.
_FOREX_TIERS = ((256, 20.0), (128, 12.0), (64, 6.0), (32, 3.2))

# LLM activation estimate: GB per sample at the reference context length, and the
# auto-batch ceiling.
_LLM_REF_SEQ_LEN = 2048
_LLM_GB_PER_REF_SAMPLE = 0.45
_LLM_MAX_BATCH = 32


@dataclass(frozen=True)
class VramProbe:
    """Where and how :func:`audit_hardware_vram` probes.

    ``res_override`` replaces the registry resolution; ``mode`` is "train" or "val";
    ``sample_fraction``, ``fold`` and ``pairs`` only label the report.
    """

    device: torch.device
    res_override: int | None = None
    mode: str = "train"
    sample_fraction: float = 1.0
    fold: str | int | None = None
    pairs: list[str] | None = None


@dataclass(frozen=True)
class _ProbeShape:
    """What the vision probe measures: model, phase, resolution and GPU size."""

    model_key: str
    mode: str
    h: int
    w: int
    is_restoration: bool
    vram_gb: float


class _HostRam(NamedTuple):
    """Host RAM in GB (64.0 when unmeasured), the Kaggle flag, and whether RAM was measured."""

    gb: float
    is_kaggle: bool
    measured: bool


def _is_ddp() -> bool:
    """Whether this process is one rank of a DDP launch."""
    return os.environ.get("RANK") is not None or int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_no_kernel_error(message: str) -> bool:
    """Whether a CUDA error message reports a missing kernel image for the GPU arch."""
    return "no kernel image is available" in message or "cudaErrorNoKernelImageForDevice" in message


# --------------------------------------------------------------------------- forex


class _DeviceReport(NamedTuple):
    """Device name, free and total memory in GB (0.0 without a device budget)."""

    name: str
    free_gb: float
    total_gb: float


def _device_report(memory: DeviceMemory) -> _DeviceReport:
    """Name and memory of the probed device, through its memory adapter."""
    budget = memory.budget()
    if budget is None:
        return _DeviceReport(memory.device_name(), 0.0, 0.0)
    return _DeviceReport(memory.device_name(), budget.free_gb, budget.total_gb)


def _forex_tier(dyn_batch: int, vram_gb: float) -> int:
    """Per-device forex batch for a dynamic batch estimate and total VRAM."""
    for batch, min_vram in _FOREX_TIERS:
        if dyn_batch >= batch and vram_gb >= min_vram:
            return batch
    return 16


def _forex_auto_batch(report: _DeviceReport, gpu_count: int, mode: str) -> int:
    """Dynamic forex batch sized from the device's free memory."""
    if report.free_gb > 0:
        # Compute dynamic batch: ~2.5MB activation footprint per multi-timeframe sample
        # Maintain 30% safety margin for optimizer momentum, activations, and device context
        usable_bytes = report.free_gb * _GIB * 0.70
        sample_bytes = 2.5 * 1024 * 1024
        per_device = _forex_tier(int(usable_bytes / sample_bytes), report.total_gb)
    else:
        per_device = 16

    is_ddp = _is_ddp()
    final_batch = per_device * gpu_count if (gpu_count > 1 and not is_ddp) else per_device
    if mode == "val":
        final_batch *= 2
    return final_batch


def _forex_batch(model_info: dict[str, Any], probe: VramProbe, memory: DeviceMemory) -> int:
    """Forex batch: the configured size, else sized from free device memory."""
    configured_batch = model_info.get("batch_size", "auto")
    report = _device_report(memory)

    if isinstance(configured_batch, int) and configured_batch > 0:
        final_batch = configured_batch if probe.mode == "train" else configured_batch * 2
    else:
        final_batch = _forex_auto_batch(report, memory.device_count(), probe.mode)

    symbols_str = " | ".join(probe.pairs) if probe.pairs else "ALL"
    fold_str = str(probe.fold) if probe.fold else "MAIN"
    logger.info(
        "Forex Dynamic Hardware Probe: %s (Free: %.1fGB / Total: %.1fGB) | Phase: %s | Batch: "
        "%d | Fold: %s | Symbols: %s",
        report.name,
        report.free_gb,
        report.total_gb,
        probe.mode.capitalize(),
        final_batch,
        fold_str,
        symbols_str,
    )
    return final_batch


# --------------------------------------------------------------------------- llm


def _is_llm(model_key: str, model_info: dict[str, Any]) -> bool:
    """Whether the model is a causal LLM / text-instruction model."""
    return (
        model_info.get("category") == "llm"
        or model_info.get("dataset_type") in ("text_instruction", "llm")
        or "llm" in model_key.lower()
    )


def _llm_auto_batch(seq_len: int, free_gb: float, mode: str) -> int:
    """Power-of-two LLM batch (1..32) sized from free VRAM and sequence length.

    Activation footprint is estimated at ~0.45 GB per 2048-token sample and scales
    linearly with the sequence length; 30% of free VRAM is kept as headroom.
    Without a device memory budget the safe default is 2 (train) / 4 (val).
    """
    if free_gb <= 0:
        return 2 if mode == "train" else 4
    sample_gb = (seq_len / _LLM_REF_SEQ_LEN) * _LLM_GB_PER_REF_SAMPLE
    usable_gb = max(0.2, free_gb * 0.70)
    dyn_batch = max(1, int(usable_gb // sample_gb))
    pow2_batch = 1 << (dyn_batch.bit_length() - 1)
    return max(1, min(pow2_batch, _LLM_MAX_BATCH))


def _llm_batch(model_info: dict[str, Any], probe: VramProbe, memory: DeviceMemory) -> int:
    """LLM batch: the configured size, else sized from free device memory and context."""
    configured_batch = model_info.get("batch_size", "auto")
    seq_len = int(probe.res_override or model_info.get("input_size") or _LLM_REF_SEQ_LEN)
    report = _device_report(memory)

    if isinstance(configured_batch, int) and configured_batch > 0:
        final_batch = configured_batch if probe.mode == "train" else configured_batch * 2
    else:
        final_batch = _llm_auto_batch(seq_len, report.free_gb, probe.mode)

    logger.info(
        "LLM Dynamic Hardware Probe: %s (Free: %.1fGB / Total: %.1fGB) | Mode: %s | "
        "Batch: %d | SeqLen: %d",
        report.name,
        report.free_gb,
        report.total_gb,
        probe.mode.capitalize(),
        final_batch,
        seq_len,
    )
    return max(1, final_batch)


# --------------------------------------------------------------------------- vision


def _is_restoration(model_key: str, model_info: dict[str, Any]) -> bool:
    """Whether the model is a heavy restoration / face / dense-prediction architecture."""
    category = str(model_info.get("category", "")).lower()
    dataset_type = str(model_info.get("dataset_type", "")).lower()
    return (
        category in ["restoration", "face"]
        or dataset_type in ["restoration", "face", "segmentation", "super_resolution"]
        or any(x in model_key.lower() for x in _RESTORATION_KEYS)
    )


def _cpu_fallback_batch(config: dict[str, Any]) -> int:
    """Configured default batch size without a device memory budget (16 when "auto"/unset)."""
    fallback_val = config.get("defaults", {}).get("batch_size", 16)
    if isinstance(fallback_val, str) and fallback_val.lower() == "auto":
        return 16
    return int(fallback_val) if fallback_val is not None else 16


def _preflight_device(cast_device: Any, memory: DeviceMemory) -> None:
    """Exit on fatal GPU faults (ECC, missing kernel image); re-raise anything else."""
    try:
        test_tensor = torch.ones(1, device=cast_device) + 1.0
        memory.synchronize()
        del test_tensor
    except Exception as exc:
        err_str = str(exc)
        if "ECC" in err_str or "uncorrectable" in err_str.lower():
            logger.critical("Uncorrectable ECC error detected on GPU memory.")
            sys.exit(1)
        if _is_no_kernel_error(err_str):
            logger.critical("No CUDA kernel image available for active GPU architecture.")
            sys.exit(1)
        raise


def _resolve_resolution(model_info: dict[str, Any], res_override: int | None) -> int:
    """Probe side length: the override, else the registry ``input_size`` (default 224)."""
    res = res_override
    if res is None:
        res_raw = model_info.get("input_size", 224)
        res = res_raw[1] if isinstance(res_raw, (list, tuple)) else res_raw
    return int(res)


def _safety_multiplier(shape: _ProbeShape) -> float:
    """Dynamic headroom tiering: the fraction of free VRAM the batch may use."""
    if shape.h >= 512 or shape.is_restoration:
        return 0.70
    if shape.h >= 384:
        return 0.75
    if shape.vram_gb < 4.5:
        return 0.80
    return 0.85


def _dummy_input(shape: _ProbeShape, cast_device: Any) -> Any:
    """One random sample at the probe resolution (dict-wrapped for diffusion models)."""
    if "diffusion" in shape.model_key.lower():
        return {"pixel_values": torch.randn(1, 3, shape.h, shape.w, device=cast_device)}
    return torch.randn(1, 3, shape.h, shape.w, device=cast_device)


def _backward_mean(output: Any) -> None:
    """Backpropagate the mean of the output (summed over tensor values of a dict output)."""
    loss = (
        sum(v.mean() for v in output.values() if isinstance(v, torch.Tensor))
        if isinstance(output, dict)
        else output.mean()
    )
    if isinstance(loss, torch.Tensor):
        loss.backward()


def _warmup_pass(
    model: nn.Module, shape: _ProbeShape, cast_device: Any, memory: DeviceMemory
) -> None:
    """One forward (and backward when training) pass to settle allocator and cuDNN state."""
    try:
        dummy = _dummy_input(shape, cast_device)
        if shape.mode == "val":
            model.eval()
        else:
            model.train()
        out = model(dummy)
        if shape.mode == "train":
            _backward_mean(out)
            model.zero_grad(set_to_none=True)
        del dummy, out
        memory.release_cache()
    except _PROBE_ERRORS as exc:
        if _is_no_kernel_error(str(exc)):
            raise
        logger.debug("Warmup pass non-fatal notification: %s", exc)


def _measure_sample_vram(
    model: nn.Module, shape: _ProbeShape, cast_device: Any, memory: DeviceMemory
) -> float:
    """Peak device bytes of one sample, measured; falls back to a resolution-scaled formula.

    Backends without allocation statistics (DirectML) measure 0 and take the formula.
    """
    memory.reset_peak()
    before_probe = memory.allocated()
    try:
        dummy_input = _dummy_input(shape, cast_device)
        if shape.mode == "train":
            model.train()
            output = model(dummy_input)
            memory.observe()
            _backward_mean(output)
            memory.observe()
            peak_probe = memory.peak_allocated()
            probe_multiplier = 1.6 if shape.is_restoration else 1.25
            sample_vram = (peak_probe - before_probe) * probe_multiplier
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
                memory.observe()
            peak_probe = memory.peak_allocated()
            val_mult = 1.6 if shape.is_restoration else 1.15
            sample_vram = (peak_probe - before_probe) * val_mult

        del dummy_input
        model.zero_grad(set_to_none=True)
        memory.release_cache()

        if sample_vram <= 0:
            raise ValueError("Probe failed to measure manifold memory footprint")
    except _PROBE_ERRORS as exc:
        logger.debug("Active probe fell back to empirical formula: %s", exc)
        res_multiplier = (shape.h * shape.w) / (224 * 224)
        base_mb = 90 if shape.is_restoration else 60
        sample_vram = base_mb * 1024 * 1024 * res_multiplier
    return sample_vram


def _pixel_cap(shape: _ProbeShape) -> int:
    """Batch cap from the per-batch pixel volume allowed for the GPU size."""
    val_mult_px = 2.0 if shape.mode == "val" else 1.0
    if shape.vram_gb < 4.5:
        max_pixels = 1.5 * (1024**2) * val_mult_px
    elif shape.vram_gb < 8.5:
        max_pixels = 5.0 * (1024**2) * val_mult_px
    elif shape.vram_gb < 16.5:
        max_pixels = 10.0 * (1024**2) * val_mult_px
    else:
        max_pixels = 32.0 * (1024**2)
    return int(max_pixels / (shape.h * shape.w))


def _host_ram() -> _HostRam:
    """Host RAM size and whether this is a Kaggle kernel."""
    is_kaggle: bool = (
        os.path.exists("/kaggle/working") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    )
    try:
        return _HostRam(psutil.virtual_memory().total / _GIB, is_kaggle, True)
    except (OSError, RuntimeError) as exc:
        logger.debug("System RAM probe fallback: %s", exc)
        return _HostRam(64.0, is_kaggle, False)


def _restoration_workspace_cap(shape: _ProbeShape, config: dict[str, Any]) -> int:
    """cuDNN workspace batch cap for restoration models (config override, else by VRAM)."""
    vram_gb = shape.vram_gb
    if shape.mode == "train":
        dynamic_cap_train = 8 if vram_gb < 8.0 else (16 if vram_gb < 16.5 else 32)
        return config.get("hardware", {}).get("cudnn_workspace_cap_train", dynamic_cap_train)
    dynamic_cap_val = (
        4 if vram_gb < 4.5 else (8 if vram_gb < 8.5 else (16 if vram_gb < 16.5 else 32))
    )
    return config.get("hardware", {}).get("cudnn_workspace_cap_val", dynamic_cap_val)


def _system_cap(shape: _ProbeShape, config: dict[str, Any], host: _HostRam) -> int:
    """Batch cap from host RAM and, for restoration models, the cuDNN workspace."""
    system_cap = 256 if shape.mode == "val" else 128
    if host.measured and (host.gb < 35.0 or host.is_kaggle):
        system_cap = min(system_cap, 32 if shape.mode == "val" else 24)
    if shape.is_restoration:
        system_cap = _restoration_workspace_cap(shape, config)
    return system_cap


def _even_batch(batch: int) -> int:
    """CuDNN bug mitigation (odd batch sizes crash ConvTranspose2d in DataParallel)."""
    if batch > 1 and batch % 2 != 0:
        return batch - 1
    return batch


def _multi_gpu_batch(final_batch: int, model: nn.Module, shape: _ProbeShape, device: Any) -> int:
    """Scale the batch for single-process multi-GPU (DataParallel or replicated) runs.

    DataParallel replication exists for CUDA (and ROCm) only; other backends count one.
    """
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    is_ddp = _is_ddp()
    is_dataparallel = isinstance(model, torch.nn.DataParallel) or (
        hasattr(model, "module") and gpu_count > 1
    )

    # In DDP each process operates on a single device, so batch must NEVER be multiplied by
    # gpu_count
    if gpu_count <= 1 or is_ddp:
        return final_batch
    if not is_dataparallel:
        return final_batch * gpu_count
    if max(shape.h, shape.w) >= 512 or shape.is_restoration:
        return min(final_batch, 4 if shape.is_restoration else 8)
    dp_factor = min(float(gpu_count), 1.5)
    return _even_batch(max(1, int(final_batch * dp_factor)))


def _restoration_vram_cap(final_batch: int, shape: _ProbeShape) -> int:
    """Cap heavy restoration / face models on <=16GB GPUs (e.g. Tesla T4).

    Physical batch size is calibrated to leave headroom for intermediate UNet activations,
    optimizer states and NCCL buffers: >= 512px capped to 2 (peak VRAM ~6.6 GiB), >= 384px
    capped to 4 (peak VRAM ~3.3 GiB, leaving >11 GiB of clean headroom).
    """
    if shape.is_restoration and shape.vram_gb < 16.5:
        if max(shape.h, shape.w) >= 512:
            return min(final_batch, 2)
        if max(shape.h, shape.w) >= 384:
            return min(final_batch, 4)
    return final_batch


def _host_val_guard(final_batch: int, shape: _ProbeShape, host: _HostRam) -> int:
    """Multi-GPU host RAM guard for validation batches."""
    if shape.mode == "val" and (host.is_kaggle or host.gb < 35.0):
        side = max(shape.h, shape.w)
        max_host_val_batch = 12 if side >= 512 else (16 if side >= 384 else 24)
        return min(final_batch, max_host_val_batch)
    return final_batch


def _check_bottleneck(final_batch: int, dynamic_batch: int, gpu_name: str, vram_gb: float) -> None:
    """Hardware bottleneck recommendation: exit when one sample cannot fit, else warn."""
    if final_batch > 1 or vram_gb >= 4.5:
        return
    if dynamic_batch < 1:
        logger.critical(
            "Hardware bottleneck reached on %s (%.1f GB). Required memory for single image "
            "exceeds available VRAM.",
            gpu_name,
            vram_gb,
        )
        sys.exit(1)
    logger.warning(
        "Hardware limit reached. Batch size forced to 1 to prevent OOM crash on %s (%.1f GB).",
        gpu_name,
        vram_gb,
    )


@dataclass(frozen=True)
class _VisionTarget:
    """The model the vision probe measures, its registry entry and the run config."""

    model_key: str
    model_info: dict[str, Any]
    config: dict[str, Any]
    model: nn.Module


def _vision_batch(target: _VisionTarget, probe: VramProbe, memory: DeviceMemory) -> int:
    """Vision / restoration batch from an empirical device-memory probe (any GPU backend)."""
    if memory.budget() is None:
        return _cpu_fallback_batch(target.config)

    cast_device: Any = probe.device
    _preflight_device(cast_device, memory)

    # Cache purge before probe
    gc.collect()
    memory.release_cache()

    budget = memory.budget()
    if budget is None:
        return _cpu_fallback_batch(target.config)
    vram_gb = budget.total_gb
    side = _resolve_resolution(target.model_info, probe.res_override)
    is_restoration = _is_restoration(target.model_key, target.model_info)
    shape = _ProbeShape(target.model_key, probe.mode, side, side, is_restoration, vram_gb)
    available_vram = budget.free_bytes * _safety_multiplier(shape)

    _warmup_pass(target.model, shape, cast_device, memory)
    sample_vram = _measure_sample_vram(target.model, shape, cast_device, memory)
    dynamic_batch = int(available_vram / sample_vram)

    host = _host_ram()
    final_batch = max(
        1, min(dynamic_batch, _pixel_cap(shape), _system_cap(shape, target.config, host))
    )
    final_batch = _even_batch(final_batch)

    gpu_name = memory.device_name()
    final_batch = _multi_gpu_batch(final_batch, target.model, shape, probe.device)
    final_batch = _restoration_vram_cap(final_batch, shape)
    final_batch = _host_val_guard(final_batch, shape, host)
    _check_bottleneck(final_batch, dynamic_batch, gpu_name, vram_gb)

    logger.info(
        "Memory Probe: %s %s (%.1f GB%s) | %s @ %dpx | Batch: %d (Pixels: %.1fM) | "
        "Fraction: %.1f%%",
        memory.name,
        gpu_name,
        vram_gb,
        "" if budget.measured else ", estimated",
        probe.mode.capitalize(),
        shape.h,
        final_batch,
        (shape.h * shape.w * final_batch) / 1e6,
        probe.sample_fraction * 100,
    )
    return final_batch


def audit_hardware_vram(
    model_key: str,
    model_info: dict[str, Any],
    config: dict[str, Any],
    model: nn.Module,
    probe: VramProbe,
) -> int:
    """
    Performs an empirical device-memory test at the specified resolution to determine
    the optimal batch size within hardware memory safety bounds.

    Device memory is read through :func:`training.hardware.device_memory.device_memory_for`
    (CUDA/ROCm, XPU, MPS, DirectML; the CPU has no budget and gets the configured default).
    """
    memory = device_memory_for(probe.device, config)
    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()
    if is_forex:
        return _forex_batch(model_info, probe, memory)
    if _is_llm(model_key, model_info):
        return _llm_batch(model_info, probe, memory)
    return _vision_batch(_VisionTarget(model_key, model_info, config, model), probe, memory)
