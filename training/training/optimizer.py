"""Optimizer and Learning Rate Scheduler Factory Builders.

Constructs parameter groups with decoupled weight decay for normalization layers
and instantiates modern schedulers with warmup and cosine decay.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable
from typing import Any

import torch
from torch import optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from training.training.param_groups import build_param_groups

logger = logging.getLogger("lemtrain.training.optimizer")

OptimizerFactory = Callable[..., optim.Optimizer]
"""``(param_groups, lr=..., betas=..., eps=...) -> optimizer``."""


def paged_adamw_8bit() -> OptimizerFactory | None:
    """``bitsandbytes.optim.PagedAdamW8bit``, or None where bitsandbytes is not installed.

    bitsandbytes is an optional, CUDA-oriented dependency (``requirements.txt``) that is
    often absent on Windows and CPU hosts, so it is imported on demand rather than at
    module import; the optimizer then falls back to ``torch.optim.AdamW``.
    """
    try:
        bnb_optim = importlib.import_module("bitsandbytes.optim")
    except ImportError:
        return None
    factory = getattr(bnb_optim, "PagedAdamW8bit", None)
    if isinstance(factory, type) and issubclass(factory, optim.Optimizer):
        return factory
    return None


def _betas(betas: Any) -> tuple[float, float]:
    """Adam betas from config; the defaults unless it is a pair."""
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        return (float(betas[0]), float(betas[1]))
    return (0.9, 0.999)


def build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
    lr: float | None = None,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    """Build the optimizer over backbone/head x decay/no-decay param groups.

    Biases and normalization layers get no weight decay; head groups are tagged so the
    governor's ``head_lr_multiplier`` reaches them (see ``param_groups``).
    """
    opt_config = config.get("optimizer", {})
    if not isinstance(opt_config, dict):
        opt_config = {}

    opt_type = opt_config.get("type", "adamw").lower()
    base_lr = float(lr if lr is not None else opt_config.get("lr", 1e-4))
    base_wd = float(
        weight_decay if weight_decay is not None else opt_config.get("weight_decay", 1e-2)
    )

    param_groups = build_param_groups(model, base_wd)
    beta_tuple = _betas(opt_config.get("betas", [0.9, 0.999]))
    eps = float(opt_config.get("eps", 1e-8))

    if opt_type == "adam":
        return optim.Adam(param_groups, lr=base_lr, betas=beta_tuple, eps=eps)
    if opt_type == "sgd":
        momentum = float(opt_config.get("momentum", 0.9))
        return optim.SGD(param_groups, lr=base_lr, momentum=momentum)
    if opt_type == "rmsprop":
        return optim.RMSprop(param_groups, lr=base_lr, eps=eps)

    # Use 8-bit Paged AdamW when bitsandbytes is available and CUDA is active
    wants_8bit = opt_type in ("paged_adamw_8bit", "adamw_8bit") or (
        opt_type == "adamw" and torch.cuda.is_available() and config.get("category") == "llm"
    )
    paged_factory = paged_adamw_8bit() if wants_8bit else None
    if paged_factory is not None:
        try:
            return paged_factory(param_groups, lr=base_lr, betas=beta_tuple, eps=eps)
        except (AttributeError, RuntimeError) as exc:
            logger.warning("Failed to initialize PagedAdamW8bit (%s); falling back.", exc)

    # Default to AdamW
    return optim.AdamW(param_groups, lr=base_lr, betas=beta_tuple, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_epochs: int,
    steps_per_epoch: int = 100,
) -> Any:
    """Build learning rate scheduler supporting Cosine, OneCycle, Step, and Warmup."""
    sched_config = config.get("scheduler", {})
    if not isinstance(sched_config, dict):
        sched_config = {}

    sched_type = sched_config.get("type", "cosine").lower()
    warmup_epochs = int(sched_config.get("warmup_epochs", 5))
    min_lr = float(sched_config.get("min_lr", 1e-6))

    if sched_type == "one_cycle":
        max_lr = [group["lr"] for group in optimizer.param_groups]
        total_steps = max(1, total_epochs * steps_per_epoch)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=float(sched_config.get("pct_start", 0.1)),
            anneal_strategy="cos",
        )

    if sched_type == "warmup_cosine":
        total_steps = max(1, total_epochs)
        warmup_steps = max(1, warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_steps:
                return float(epoch + 1) / float(warmup_steps)
            progress = float(epoch - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if sched_type == "step":
        return StepLR(
            optimizer,
            step_size=int(sched_config.get("step_size", 30)),
            gamma=float(sched_config.get("gamma", 0.1)),
        )

    if sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get("mode", "min"),
            factor=float(sched_config.get("factor", 0.5)),
            patience=int(sched_config.get("patience", 5)),
            min_lr=min_lr,
        )

    if sched_type == "restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sched_config.get("t_0", 10)),
            T_mult=int(sched_config.get("t_mult", 2)),
            eta_min=min_lr,
        )

    # Default: Standard Cosine Annealing
    return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=min_lr)
