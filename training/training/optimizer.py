"""Optimizer and Learning Rate Scheduler Factory Builders.

Constructs parameter groups with decoupled weight decay for normalization layers
and instantiates modern schedulers with warmup and cosine decay.
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
    lr: float | None = None,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    """Build optimizer with decoupled weight decay for biases and normalization layers."""
    opt_config = config.get("optimizer", {})
    if not isinstance(opt_config, dict):
        opt_config = {}

    opt_type = opt_config.get("type", "adamw").lower()
    base_lr = float(lr if lr is not None else opt_config.get("lr", 1e-4))
    base_wd = float(weight_decay if weight_decay is not None else opt_config.get("weight_decay", 1e-2))

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": base_wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    betas = opt_config.get("betas", [0.9, 0.999])
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        beta_tuple = (float(betas[0]), float(betas[1]))
    else:
        beta_tuple = (0.9, 0.999)

    eps = float(opt_config.get("eps", 1e-8))

    if opt_type == "adam":
        return optim.Adam(param_groups, lr=base_lr, betas=beta_tuple, eps=eps)
    if opt_type == "sgd":
        momentum = float(opt_config.get("momentum", 0.9))
        return optim.SGD(param_groups, lr=base_lr, momentum=momentum)
    if opt_type == "rmsprop":
        return optim.RMSprop(param_groups, lr=base_lr, eps=eps)

    # Default to AdamW
    return optim.AdamW(param_groups, lr=base_lr, betas=beta_tuple, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    total_epochs: int,
    steps_per_epoch: int = 100,
    reset_scheduler: bool = False,
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
        step_size = int(sched_config.get("step_size", 30))
        gamma = float(sched_config.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if sched_type == "plateau":
        mode = sched_config.get("mode", "min")
        factor = float(sched_config.get("factor", 0.5))
        patience = int(sched_config.get("patience", 5))
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    if sched_type == "restarts":
        t_0 = int(sched_config.get("t_0", 10))
        t_mult = int(sched_config.get("t_mult", 2))
        return CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=min_lr)

    # Default: Standard Cosine Annealing
    return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=min_lr)
