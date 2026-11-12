"""Governor learning-rate multipliers composed with the LR scheduler.

The governor cools or jolts the learning rate with a multiplier. Applying it by overwriting
``param_group["lr"]`` with ``initial_lr * multiplier`` erased the scheduler's decay, and
the scheduler's next step overwrote the multiplier again. Instead, the change in multiplier
is applied as a ratio to every learning-rate value the optimizer group and the scheduler
derive later rates from, so decay and governor scaling compose multiplicatively.

The applied multiplier is stored in each param group (``APPLIED_MULT_KEY``) and therefore
survives an optimizer ``state_dict`` round trip on resume.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("lemtrain.training.lr_scaling")

APPLIED_MULT_KEY = "governor_lr_mult"
"""Param-group key holding the governor multiplier already applied to the group."""

GROUP_ROLE_KEY = "group"
"""Param-group key naming its role; ``"head"`` groups follow ``head_lr_multiplier``."""

HEAD_ROLE = "head"

# Param-group values schedulers derive learning rates from (OneCycleLR uses all of them).
_LR_KEYS = ("lr", "initial_lr", "max_lr", "min_lr")


def group_multiplier(group: dict[str, Any], backbone_mult: float, head_mult: float) -> float:
    """Governor multiplier for ``group``: the head multiplier for head groups only."""
    return head_mult if group.get(GROUP_ROLE_KEY) == HEAD_ROLE else backbone_mult


def scale_learning_rates(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    backbone_mult: float,
    head_mult: float,
) -> None:
    """Move every group from its applied multiplier to the governor's current one.

    Rates already decayed by the scheduler keep their decay; only the ratio between the
    new and the previously applied multiplier is applied, to the group's rates and to the
    scheduler's ``base_lrs``.
    """
    base_lrs = getattr(scheduler, "base_lrs", None)
    for index, group in enumerate(optimizer.param_groups):
        target = group_multiplier(group, backbone_mult, head_mult)
        applied = float(group.get(APPLIED_MULT_KEY, 1.0))
        if target <= 0.0 or applied <= 0.0:
            logger.warning(
                "Ignoring non-positive LR multiplier (applied %.4f, target %.4f).", applied, target
            )
            continue
        ratio = target / applied
        for key in _LR_KEYS:
            if key in group:
                group[key] = float(group[key]) * ratio
        if isinstance(base_lrs, list) and index < len(base_lrs):
            base_lrs[index] = float(base_lrs[index]) * ratio
        group[APPLIED_MULT_KEY] = target
