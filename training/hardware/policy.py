"""Execution policy, AMP, cuDNN, and memory format configuration for LemGendary Model Training Suite."""

from dataclasses import dataclass
import logging
import torch
from typing import Any

from training.hardware.discovery import DeviceInfo

logger = logging.getLogger("lemtrain.hardware.policy")


@dataclass(frozen=True)
class HardwarePolicyResult:
    """Configured execution parameters and mixed-precision state."""
    amp_enabled: bool
    amp_dtype: torch.dtype
    cudnn_benchmark: bool
    tf32_enabled: bool
    channels_last: bool
    scaler: Any | None


ExecutionPolicy = HardwarePolicyResult


def apply_hardware_policy(
    model_key: str,
    model_info: dict[str, Any],
    device_info: DeviceInfo,
    config: dict[str, Any] | None = None,
) -> HardwarePolicyResult:
    """
    Apply global execution and mixed-precision policies based on accelerator capabilities and task domain.
    
    Forex tasks disable AMP by default due to high dynamic range / overflow risks in return distributions.
    Vision and restoration models on CUDA sm_70+ utilize AMP FP16 and cuDNN benchmarking.
    """
    cfg = config or {}
    hw_cfg = cfg.get("hardware", {})

    is_cuda = device_info.is_cuda
    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()

    # 1. cuDNN Benchmark
    cudnn_benchmark = False
    if is_cuda:
        cudnn_benchmark = hw_cfg.get("cudnn_benchmark", True)
        torch.backends.cudnn.benchmark = cudnn_benchmark

    # 2. TensorFloat-32 (TF32) on Ampere+ (sm_80+)
    tf32_enabled = False
    if is_cuda and device_info.capability[0] >= 8:
        tf32_enabled = hw_cfg.get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    # 3. Mixed-Precision (AMP) Policy
    amp_enabled = False
    amp_dtype = torch.float16

    if is_forex:
        # Forex gradients and cumulative returns overflow under FP16
        amp_enabled = False
        logger.info("AMP disabled for Forex model '%s' (FP16 overflow prevention).", model_key)
    elif is_cuda:
        amp_enabled = hw_cfg.get("amp_enabled", True)
        # Check for BF16 support on sm_80+
        use_bf16 = hw_cfg.get("use_bf16", False) and device_info.capability[0] >= 8 and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 4. GradScaler
    scaler = None
    if is_cuda and amp_enabled and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 5. Channels-Last Memory Format (optimal on Tensor Cores for 4D vision tensors)
    channels_last = False
    if is_cuda and not is_forex and device_info.capability[0] >= 7:
        channels_last = hw_cfg.get("channels_last", True)

    logger.debug(
        "Hardware policy applied for %s: AMP=%s (%s), cuDNN Benchmark=%s, TF32=%s, Channels-Last=%s",
        model_key, amp_enabled, amp_dtype, cudnn_benchmark, tf32_enabled, channels_last
    )

    return HardwarePolicyResult(
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        cudnn_benchmark=cudnn_benchmark,
        tf32_enabled=tf32_enabled,
        channels_last=channels_last,
        scaler=scaler,
    )
