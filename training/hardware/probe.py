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


def _gpu0_free_gb() -> float:
    """Free VRAM on GPU 0 in GB (0.0 without CUDA or when the query fails)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_vram_probe, _ = torch.cuda.mem_get_info(0)
        return free_vram_probe / _GIB
    except RuntimeError as mem_err:
        logger.debug("torch.cuda.mem_get_info unavailable; assuming 0 GB free: %s", mem_err)
        return 0.0


def _forex_tier(dyn_batch: int, vram_gb: float) -> int:
    """Per-device forex batch for a dynamic batch estimate and total VRAM."""
    for batch, min_vram in _FOREX_TIERS:
        if dyn_batch >= batch and vram_gb >= min_vram:
            return batch
    return 16


def _forex_auto_batch(vram_gb: float, mode: str) -> tuple[int, float]:
    """Dynamic forex batch and the free VRAM (GB) it was sized from."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if torch.cuda.is_available():
        free_vram, _ = torch.cuda.mem_get_info(0)
        free_gb = free_vram / _GIB
        # Compute dynamic batch: ~2.5MB activation footprint per multi-timeframe sample
        # Maintain 30% safety margin for optimizer momentum, activations, and CUDA context
        usable_bytes = free_vram * 0.70
        sample_bytes = 2.5 * 1024 * 1024
        per_device = _forex_tier(int(usable_bytes / sample_bytes), vram_gb)
    else:
        free_gb = 0.0
        per_device = 16

    is_ddp = _is_ddp()
    final_batch = per_device * gpu_count if (gpu_count > 1 and not is_ddp) else per_device
    if mode == "val":
        final_batch *= 2
    return final_batch, free_gb


def _forex_batch(model_info: dict[str, Any], probe: VramProbe) -> int:
    """Forex batch: the configured size, else sized from free VRAM."""
    configured_batch = model_info.get("batch_size", "auto")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        (torch.cuda.get_device_properties(0).total_memory / _GIB)
        if torch.cuda.is_available()
        else 0.0
    )
    free_gb = _gpu0_free_gb()

    if isinstance(configured_batch, int) and configured_batch > 0:
        final_batch = configured_batch if probe.mode == "train" else configured_batch * 2
    else:
        final_batch, free_gb = _forex_auto_batch(vram_gb, probe.mode)

    symbols_str = " | ".join(probe.pairs) if probe.pairs else "ALL"
    fold_str = str(probe.fold) if probe.fold else "MAIN"
    logger.info(
        "Forex Dynamic Hardware Probe: %s (Free: %.1fGB / Total: %.1fGB) | Phase: %s | Batch: "
        "%d | Fold: %s | Symbols: %s",
        gpu_name,
        free_gb if torch.cuda.is_available() else 0.0,
        vram_gb,
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
    Without CUDA (or a free-VRAM reading) the safe default is 2 (train) / 4 (val).
    """
    if not torch.cuda.is_available() or free_gb <= 0:
        return 2 if mode == "train" else 4
    sample_gb = (seq_len / _LLM_REF_SEQ_LEN) * _LLM_GB_PER_REF_SAMPLE
    usable_gb = max(0.2, free_gb * 0.70)
    dyn_batch = max(1, int(usable_gb // sample_gb))
    pow2_batch = 1 << (dyn_batch.bit_length() - 1)
    return max(1, min(pow2_batch, _LLM_MAX_BATCH))


def _llm_batch(model_info: dict[str, Any], probe: VramProbe) -> int:
    """LLM batch: the configured size, else sized from free VRAM and context length."""
    configured_batch = model_info.get("batch_size", "auto")
    seq_len = int(probe.res_override or model_info.get("input_size") or _LLM_REF_SEQ_LEN)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        (torch.cuda.get_device_properties(0).total_memory / _GIB)
        if torch.cuda.is_available()
        else 0.0
    )
    free_gb = _gpu0_free_gb()

    if isinstance(configured_batch, int) and configured_batch > 0:
        final_batch = configured_batch if probe.mode == "train" else configured_batch * 2
    else:
        final_batch = _llm_auto_batch(seq_len, free_gb, probe.mode)

    logger.info(
        "LLM Dynamic Hardware Probe: %s (Free: %.1fGB / Total: %.1fGB) | Mode: %s | "
        "Batch: %d | SeqLen: %d",
        gpu_name,
        free_gb,
        vram_gb,
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
    """Configured default batch size for non-CUDA devices (16 when "auto" or unset)."""
    fallback_val = config.get("defaults", {}).get("batch_size", 16)
    if isinstance(fallback_val, str) and fallback_val.lower() == "auto":
        return 16
    return int(fallback_val) if fallback_val is not None else 16


def _preflight_cuda(cast_device: Any) -> None:
    """Exit on fatal GPU faults (ECC, missing kernel image); re-raise anything else."""
    try:
        test_tensor = torch.ones(1, device=cast_device) + 1.0
        torch.cuda.synchronize()
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


def _free_vram() -> tuple[int, float]:
    """Free VRAM bytes on GPU 0 (incl. unused reserved cache) and its total size in GB."""
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    vram_gb = total_vram / _GIB
    unused_reserved = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    return free_vram + unused_reserved, vram_gb


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


def _warmup_pass(model: nn.Module, shape: _ProbeShape, cast_device: Any) -> None:
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
        torch.cuda.empty_cache()
    except _PROBE_ERRORS as exc:
        if _is_no_kernel_error(str(exc)):
            raise
        logger.debug("Warmup pass non-fatal notification: %s", exc)


def _measure_sample_vram(model: nn.Module, shape: _ProbeShape, cast_device: Any) -> float:
    """Peak VRAM bytes of one sample, measured; falls back to a resolution-scaled formula."""
    torch.cuda.reset_peak_memory_stats(0)
    before_probe = torch.cuda.memory_allocated(0)
    try:
        dummy_input = _dummy_input(shape, cast_device)
        if shape.mode == "train":
            model.train()
            output = model(dummy_input)
            _backward_mean(output)
            peak_probe = torch.cuda.max_memory_allocated(0)
            probe_multiplier = 1.6 if shape.is_restoration else 1.25
            sample_vram = (peak_probe - before_probe) * probe_multiplier
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_probe = torch.cuda.max_memory_allocated(0)
            val_mult = 1.6 if shape.is_restoration else 1.15
            sample_vram = (peak_probe - before_probe) * val_mult

        del dummy_input
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

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
    """Scale the batch for single-process multi-GPU (DataParallel or replicated) runs."""
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


def _vision_batch(
    model_key: str,
    model_info: dict[str, Any],
    config: dict[str, Any],
    model: nn.Module,
    probe: VramProbe,
) -> int:
    """Vision / restoration batch from an empirical VRAM probe (full 24-model coverage)."""
    if probe.device.type != "cuda":
        return _cpu_fallback_batch(config)

    cast_device: Any = probe.device
    _preflight_cuda(cast_device)

    # Cache purge before probe
    gc.collect()
    torch.cuda.empty_cache()

    free_vram, vram_gb = _free_vram()
    side = _resolve_resolution(model_info, probe.res_override)
    shape = _ProbeShape(
        model_key, probe.mode, side, side, _is_restoration(model_key, model_info), vram_gb
    )
    available_vram = free_vram * _safety_multiplier(shape)

    _warmup_pass(model, shape, cast_device)
    dynamic_batch = int(available_vram / _measure_sample_vram(model, shape, cast_device))

    host = _host_ram()
    final_batch = max(1, min(dynamic_batch, _pixel_cap(shape), _system_cap(shape, config, host)))
    final_batch = _even_batch(final_batch)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    final_batch = _multi_gpu_batch(final_batch, model, shape, probe.device)
    final_batch = _restoration_vram_cap(final_batch, shape)
    final_batch = _host_val_guard(final_batch, shape, host)
    _check_bottleneck(final_batch, dynamic_batch, gpu_name, vram_gb)

    logger.info(
        "Memory Probe: %s (%.1f GB) | %s @ %dpx | Batch: %d (Pixels: %.1fM) | Fraction: %.1f%%",
        gpu_name,
        vram_gb,
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
    Performs an empirical VRAM test at the specified resolution to determine
    the optimal batch size within hardware memory safety bounds.
    """
    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()
    if is_forex:
        return _forex_batch(model_info, probe)
    if _is_llm(model_key, model_info):
        return _llm_batch(model_info, probe)
    return _vision_batch(model_key, model_info, config, model, probe)
