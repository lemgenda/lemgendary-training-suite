"""Readers of the ``optimizer`` section of ``config.yaml`` (the one place for its defaults).

``build_optimizer`` builds from these values, the gradient clip of the train step reads
:func:`grad_clip_norm`, and the run-settings record (``training.run_settings``) stores
them. This module is torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OPTIMIZER_SECTION = "optimizer"
DEFAULT_OPTIMIZER_TYPE = "adamw"
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
DEFAULT_MOMENTUM = 0.9
DEFAULT_GRAD_CLIP = 1.0
"""Max global gradient norm of the train step (``optimizer.grad_clip``; 0 turns it off)."""


def optimizer_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The ``optimizer`` section, empty when missing or not a mapping."""
    section = (config or {}).get(OPTIMIZER_SECTION)
    return section if isinstance(section, Mapping) else {}


def optimizer_type(config: Mapping[str, Any] | None) -> str:
    """The configured optimizer type, lower case (``adamw`` by default)."""
    return str(optimizer_section(config).get("type", DEFAULT_OPTIMIZER_TYPE)).lower()


def base_learning_rate(config: Mapping[str, Any] | None, lr: float | None = None) -> float:
    """``lr`` (the ``--lr`` override) when given, else ``optimizer.lr``."""
    return float(lr if lr is not None else optimizer_section(config).get("lr", DEFAULT_LR))


def base_weight_decay(config: Mapping[str, Any] | None, weight_decay: float | None = None) -> float:
    """``weight_decay`` when given, else ``optimizer.weight_decay``."""
    if weight_decay is not None:
        return float(weight_decay)
    return float(optimizer_section(config).get("weight_decay", DEFAULT_WEIGHT_DECAY))


def adam_betas(config: Mapping[str, Any] | None) -> tuple[float, float]:
    """``optimizer.betas`` when it is a pair, else the Adam defaults."""
    betas = optimizer_section(config).get("betas", list(DEFAULT_BETAS))
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        return (float(betas[0]), float(betas[1]))
    return DEFAULT_BETAS


def optimizer_eps(config: Mapping[str, Any] | None) -> float:
    """``optimizer.eps``."""
    return float(optimizer_section(config).get("eps", DEFAULT_EPS))


def sgd_momentum(config: Mapping[str, Any] | None) -> float:
    """``optimizer.momentum`` (SGD only)."""
    return float(optimizer_section(config).get("momentum", DEFAULT_MOMENTUM))


def grad_clip_norm(config: Mapping[str, Any] | None) -> float:
    """``optimizer.grad_clip``: the max gradient norm of a train step (0 turns clipping off).

    Raises:
        ValueError: the value is negative.
    """
    value = float(optimizer_section(config).get("grad_clip", DEFAULT_GRAD_CLIP))
    if value < 0.0:
        raise ValueError(f"optimizer.grad_clip must be >= 0, got {value}")
    return value


def optimizer_shape(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Optimizer type and the hyperparameters that shape its state (fixed for a run)."""
    return {
        "type": optimizer_type(config),
        "betas": list(adam_betas(config)),
        "eps": optimizer_eps(config),
        "momentum": sgd_momentum(config),
    }
