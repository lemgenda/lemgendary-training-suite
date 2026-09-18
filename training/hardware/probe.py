"""Hardware VRAM probing and dynamic batch-size calculation for LemGendary Model Training Suite."""

import gc
import logging
import os
import sys
import torch
import torch.nn as nn
from typing import Any

logger = logging.getLogger("lemtrain.hardware.probe")


def audit_hardware_vram(
    model_key: str,
    model_info: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    model: nn.Module,
    res_override: int | None = None,
    mode: str = "train",
    sample_fraction: float = 1.0,
    fold: str | int | None = None,
    pairs: list[str] | None = None,
) -> int:
    """
    Performs an empirical VRAM test at the specified resolution to determine
    the optimal batch size within hardware memory safety bounds.
    """
    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()

    if is_forex:
        configured_batch = model_info.get("batch_size", "auto")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0

        if isinstance(configured_batch, int) and configured_batch > 0:
            final_batch = configured_batch if mode == "train" else configured_batch * 2
        else:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if vram_gb >= 20.0:
                per_device = 256
            elif vram_gb >= 14.0:
                per_device = 128
            elif vram_gb >= 7.0:
                per_device = 64
            elif vram_gb >= 3.5:
                per_device = 32
            else:
                per_device = 16

            final_batch = per_device * gpu_count if gpu_count > 1 else per_device
            if mode == "val":
                final_batch *= 2

        symbols_str = " | ".join(pairs) if pairs else "ALL"
        fold_str = str(fold) if fold else "MAIN"
        logger.info(
            "Forex Memory Probe: %s (%.1fGB) | Phase: %s | Batch: %d | Fold: %s | Symbols: %s",
            gpu_name, vram_gb, mode.capitalize(), final_batch, fold_str, symbols_str
        )
        return final_batch

    # Vision / Restoration Models
    is_restoration = any(
        x in model_key.lower()
        for x in ["nafnet", "mprnet", "mirnet", "ffanet", "codeformer", "film_restorer", "parsenet"]
    )

    if device.type != "cuda":
        fallback_val = config.get("defaults", {}).get("batch_size", 16)
        if isinstance(fallback_val, str) and fallback_val.lower() == "auto":
            return 16
        return int(fallback_val) if fallback_val is not None else 16

    # Pre-flight CUDA check
    try:
        test_tensor = torch.ones(1, device=device) + 1.0
        torch.cuda.synchronize()
        del test_tensor
    except Exception as exc:
        err_str = str(exc)
        if "ECC" in err_str or "uncorrectable" in err_str.lower():
            logger.critical("Uncorrectable ECC error detected on GPU memory.")
            sys.exit(1)
        if "no kernel image is available" in err_str or "cudaErrorNoKernelImageForDevice" in err_str:
            logger.critical("No CUDA kernel image available for active GPU architecture.")
            sys.exit(1)
        raise exc

    # Cache purge before probe
    gc.collect()
    torch.cuda.empty_cache()

    free_vram, total_vram = torch.cuda.mem_get_info(0)
    vram_gb = total_vram / (1024**3)
    unused_reserved = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    free_vram = free_vram + unused_reserved

    is_exhausted = (free_vram / total_vram) < 0.15

    res = res_override
    if res is None:
        res_raw = model_info.get("input_size", 224)
        res = res_raw[1] if isinstance(res_raw, (list, tuple)) else res_raw
    h = w = int(res)

    # Dynamic Headroom Tiering
    if h >= 512 or is_restoration:
        safety_multiplier = 0.70
    elif h >= 384:
        safety_multiplier = 0.75
    elif vram_gb < 4.5:
        safety_multiplier = 0.80
    else:
        safety_multiplier = 0.85

    available_vram = free_vram * safety_multiplier

    # Warmup pass
    try:
        if "diffusion" in model_key.lower():
            dummy = {"pixel_values": torch.randn(1, 3, h, w, device=device)}
        else:
            dummy = torch.randn(1, 3, h, w, device=device)

        model.eval() if mode == "val" else model.train()
        out = model(dummy)
        if mode == "train":
            loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor)) if isinstance(out, dict) else out.mean()
            if isinstance(loss, torch.Tensor):
                loss.backward()
            model.zero_grad(set_to_none=True)
        del dummy, out
        torch.cuda.empty_cache()
    except Exception as exc:
        err_msg = str(exc)
        if "no kernel image is available" in err_msg or "cudaErrorNoKernelImageForDevice" in err_msg:
            raise
        logger.debug("Warmup pass non-fatal notification: %s", exc)

    torch.cuda.reset_peak_memory_stats(0)
    before_probe = torch.cuda.memory_allocated(0)

    try:
        if "diffusion" in model_key.lower():
            dummy_input: Any = {"pixel_values": torch.randn(1, 3, h, w, device=device)}
        else:
            dummy_input = torch.randn(1, 3, h, w, device=device)

        if mode == "train":
            model.train()
            output = model(dummy_input)
            loss = sum(v.mean() for v in output.values() if isinstance(v, torch.Tensor)) if isinstance(output, dict) else output.mean()
            if isinstance(loss, torch.Tensor):
                loss.backward()
            peak_probe = torch.cuda.max_memory_allocated(0)
            probe_multiplier = 1.6 if is_restoration else 1.25
            sample_vram = (peak_probe - before_probe) * probe_multiplier
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_probe = torch.cuda.max_memory_allocated(0)
            val_mult = 1.6 if is_restoration else 1.15
            sample_vram = (peak_probe - before_probe) * val_mult

        del dummy_input
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        if sample_vram <= 0:
            raise ValueError("Probe failed to measure manifold memory footprint")
    except Exception as exc:
        logger.debug("Active probe fell back to empirical formula: %s", exc)
        res_multiplier = (h * w) / (224 * 224)
        base_mb = 90 if is_restoration else 60
        sample_vram = base_mb * 1024 * 1024 * res_multiplier

    dynamic_batch = int(available_vram / sample_vram)

    # Pixel Volume Cap
    val_mult_px = 2.0 if mode == "val" else 1.0
    if vram_gb < 4.5:
        max_pixels = 1.5 * (1024**2) * val_mult_px
    elif vram_gb < 8.5:
        max_pixels = 5.0 * (1024**2) * val_mult_px
    elif vram_gb < 16.5:
        max_pixels = 10.0 * (1024**2) * val_mult_px
    else:
        max_pixels = 32.0 * (1024**2)

    pixel_cap = int(max_pixels / (h * w))
    system_cap = 256 if mode == "val" else 128

    # System RAM safeguard
    sys_ram_gb: float = 64.0
    is_kaggle: bool = os.path.exists("/kaggle/working") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    try:
        import psutil
        sys_ram_gb = psutil.virtual_memory().total / (1024**3)
        if sys_ram_gb < 35.0 or is_kaggle:
            system_cap = min(system_cap, 32 if mode == "val" else 24)
    except Exception as exc:
        logger.debug("System RAM probe fallback: %s", exc)

    if is_restoration:
        if mode == "train":
            dynamic_cap_train = 8 if vram_gb < 8.0 else (16 if vram_gb < 16.5 else 32)
            system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_train", dynamic_cap_train)
        else:
            dynamic_cap_val = 4 if vram_gb < 4.5 else (8 if vram_gb < 8.5 else (16 if vram_gb < 16.5 else 32))
            system_cap = config.get("hardware", {}).get("cudnn_workspace_cap_val", dynamic_cap_val)

    final_batch = max(1, min(dynamic_batch, pixel_cap, system_cap))

    # CuDNN bug mitigation (Odd batch sizes crash ConvTranspose2d in DataParallel)
    if final_batch > 1 and final_batch % 2 != 0:
        final_batch -= 1

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    is_dataparallel = isinstance(model, torch.nn.DataParallel) or (hasattr(model, "module") and gpu_count > 1)

    if gpu_count > 1:
        if is_dataparallel:
            if max(h, w) >= 512 or is_restoration:
                final_batch = min(final_batch, 4 if is_restoration else 8)
            else:
                dp_factor = min(float(gpu_count), 1.5)
                final_batch = max(1, int(final_batch * dp_factor))
                if final_batch > 1 and final_batch % 2 != 0:
                    final_batch -= 1
        else:
            final_batch = final_batch * gpu_count

    # Multi-GPU Host RAM Guard
    if mode == "val" and (is_kaggle or sys_ram_gb < 35.0):
        max_host_val_batch = 12 if max(h, w) >= 512 else (16 if max(h, w) >= 384 else 24)
        final_batch = min(final_batch, max_host_val_batch)

    # Hardware Bottleneck Recommendation
    if final_batch <= 1 and vram_gb < 4.5:
        if dynamic_batch < 1:
            logger.critical(
                "Hardware bottleneck reached on %s (%.1f GB). Required memory for single image exceeds available VRAM.",
                gpu_name, vram_gb
            )
            sys.exit(1)
        else:
            logger.warning(
                "Hardware limit reached. Batch size forced to 1 to prevent OOM crash on %s (%.1f GB).",
                gpu_name, vram_gb
            )

    logger.info(
        "Memory Probe: %s (%.1f GB) | %s @ %dpx | Batch: %d (Pixels: %.1fM) | Fraction: %.1f%%",
        gpu_name, vram_gb, mode.capitalize(), h, final_batch, (h * w * final_batch) / 1e6, sample_fraction * 100
    )
    return final_batch

