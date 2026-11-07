"""Pareto Model Soup and Stochastic Weight Averaging (SWA) weight blending service.

Implements model parameter interpolation across Pareto-optimal vault checkpoints
(e.g., balancing PSNR reconstruction fidelity with LPIPS perceptual detail)
based on Model Soups (Wortsman et al., 2022).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("lemtrain.export.soup")


def _normalize_weights(n: int, weights: list[float] | None) -> list[float]:
    """Return normalized blending coefficients summing to 1.0."""
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(
            f"Weights count ({len(weights)}) does not match state_dicts count ({n})."
        )
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")
    return [w / total_w for w in weights]


def blend_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute weighted linear model soup average across a list of model state dicts.

    Args:
        state_dicts: List of state dictionaries sharing the exact same architecture.
        weights: Optional blending coefficients summing to 1.0. If omitted, uniform
            averaging is applied.

    Returns:
        New state dictionary containing the interpolated model parameters.
    """
    if not state_dicts:
        raise ValueError("Cannot blend empty state_dicts list.")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    coeffs = _normalize_weights(len(state_dicts), weights)
    base_keys = state_dicts[0].keys()
    blended: dict[str, torch.Tensor] = {}

    for k in base_keys:
        ref_tensor = state_dicts[0][k]
        if not ref_tensor.is_floating_point():
            # Non-float parameters (e.g. integer steps, buffers) are inherited from first state dict
            blended[k] = ref_tensor.clone()
            continue

        accum = torch.zeros_like(ref_tensor, dtype=torch.float32)
        for sd, coeff in zip(state_dicts, coeffs):
            accum += sd[k].to(torch.float32) * coeff
        blended[k] = accum.to(ref_tensor.dtype)

    return blended


def create_pareto_model_soup(
    checkpoint_paths: list[Path],
    weights: list[float] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Load, blend, and optionally save a Pareto Model Soup checkpoint from vault binaries.

    Args:
        checkpoint_paths: Paths to input checkpoint files (e.g. [vault_psnr.pth, vault_lpips.pth]).
        weights: Blending coefficients (e.g. [0.5, 0.5]).
        output_path: Path where the blended checkpoint should be written.

    Returns:
        The full blended checkpoint payload dictionary.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for Pareto Model Soup.")

    valid_paths = [p for p in checkpoint_paths if p.exists() and p.is_file()]
    if not valid_paths:
        raise FileNotFoundError(f"None of the provided checkpoint paths exist: {checkpoint_paths}")

    loaded_payloads: list[dict[str, Any]] = []
    state_dicts: list[dict[str, torch.Tensor]] = []

    for p in valid_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        loaded_payloads.append(ckpt)
        sd = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if not isinstance(sd, dict):
            raise KeyError(f"Checkpoint at {p} does not contain valid model_state dictionary.")
        state_dicts.append(sd)

    coeffs = _normalize_weights(len(valid_paths), weights)
    blended_sd = blend_state_dicts(state_dicts, weights=coeffs)

    # Use first checkpoint as template for metadata payload, stripping single-donor optimizer
    # and scheduler state so resuming from a soup checkpoint starts with a clean optimizer.
    out_payload = dict(loaded_payloads[0])
    for stale_key in (
        "optimizer_state",
        "optimizer_state_dict",
        "scheduler_state",
        "scheduler_state_dict",
    ):
        out_payload.pop(stale_key, None)
    out_payload["model_state"] = blended_sd
    out_payload["is_pareto_soup"] = True
    out_payload["soup_donors"] = [p.name for p in valid_paths]
    out_payload["soup_weights"] = coeffs

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_payload, output_path)
        logger.info("Successfully exported Pareto Model Soup to %s", output_path)

    return out_payload
