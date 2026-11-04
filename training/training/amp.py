"""Automatic Mixed Precision (AMP) and numerical stabilization module.

Manages autocast contexts, gradient scalers, gradient unscaling and clipping,
loss accumulation scaling, and GPU-accelerated tensor metrics.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from training.hardware.policy import ExecutionPolicy


def create_grad_scaler(policy: ExecutionPolicy) -> Any | None:
    """Create a GradScaler if AMP is enabled and the policy requests it."""
    if getattr(policy, "scaler", None) is not None:
        return policy.scaler
    if getattr(policy, "amp_enabled", False) or getattr(policy, "use_amp", False):
        return torch.amp.GradScaler("cuda")
    return None


def get_autocast_context(policy: ExecutionPolicy, device: torch.device) -> Any:
    """Return appropriate autocast context manager based on device and execution policy."""
    amp_enabled = getattr(policy, "amp_enabled", False) or getattr(policy, "use_amp", False)
    if amp_enabled and device.type == "cuda":
        amp_dtype = getattr(policy, "amp_dtype", torch.float16)
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


@dataclass(frozen=True)
class GradClip:
    """Gradient clipping of one :func:`safe_backward` step.

    Attributes:
        model: Model containing parameters.
        max_norm: Maximum gradient norm for clipping.
        should_unscale: Whether to unscale gradients and clip norm on this step.
            When using gradient accumulation, this must be False until the optimizer
            update step boundary to comply with PyTorch GradScaler constraints.
    """

    model: torch.nn.Module | None = None
    max_norm: float | None = None
    should_unscale: bool = True


def safe_backward(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None = None,
    clip: GradClip | None = None,
) -> float:
    """Perform a backward pass with optional AMP scaling, gradient unscaling, and clipping.

    Args:
        loss: Scaled loss tensor.
        optimizer: PyTorch optimizer.
        scaler: GradScaler instance (if AMP active).
        clip: Gradient clipping; ``None`` skips unscaling and clipping.

    Returns:
        float: Gradient norm after clipping (or 0.0 if not computed/deferred).
    """
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    clip = clip or GradClip()
    model, max_norm, should_unscale = clip.model, clip.max_norm, clip.should_unscale
    grad_norm = 0.0
    if should_unscale and model is not None and max_norm is not None and max_norm > 0:
        if scaler is not None:
            # Check if this optimizer has already been unscaled in this step
            try:
                scaler.unscale_(optimizer)
            except RuntimeError as err:
                if "already been called" not in str(err):
                    raise
        norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        if isinstance(norm_tensor, torch.Tensor):
            grad_norm = float(norm_tensor.item())
        elif isinstance(norm_tensor, (int, float)):
            grad_norm = float(norm_tensor)

    return grad_norm


def _gaussian_kernel(
    window_size: int, sigma: float, channel: int, device: torch.device
) -> torch.Tensor:
    """Normalised 2-D Gaussian window, one copy per channel: ``[C, 1, k, k]``."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = (gauss / gauss.sum()).unsqueeze(1)

    kernel_2d = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


def _local_moments(
    img1: torch.Tensor, img2: torch.Tensor, kernel: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Windowed ``mu1^2, mu2^2, mu1*mu2, sigma1^2, sigma2^2, sigma12`` for SSIM."""
    channel, pad = kernel.size(0), kernel.size(-1) // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=channel)

    mu1 = blur(img1)
    mu2 = blur(img2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    return (
        mu1_sq,
        mu2_sq,
        mu1_mu2,
        blur(img1 * img1) - mu1_sq,
        blur(img2 * img2) - mu2_sq,
        blur(img1 * img2) - mu1_mu2,
    )


def compute_ssim_gpu(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute GPU-accelerated vectorized Structural Similarity Index (SSIM)."""
    kernel = _gaussian_kernel(window_size, sigma, img1.size(1), img1.device)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = _local_moments(img1, img2, kernel)

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()
