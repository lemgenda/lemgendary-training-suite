"""Optimizer param groups (backbone/head x decay/no-decay) and legacy-state migration.

The governor sets a separate ``head_lr_multiplier`` (head jolts, cooling the backbone
while the head keeps learning), and :func:`training.training.lr_scaling.scale_learning_rates`
applies it to param groups tagged ``group: "head"``. The optimizer used to build only a
decay and a no-decay group with no head tag, so every head multiplier was a no-op.

:func:`build_param_groups` splits the trainable parameters into up to four groups, in
this order: backbone decay, backbone no-decay, head decay, head no-decay (empty head
groups are left out). The head is named by the model's ``head_param_prefixes``
attribute when it has one; otherwise it is the last top-level child module that holds
trainable parameters, provided the model has at least two such children.

Checkpoints written before the split hold the two legacy groups.
:func:`migrate_legacy_optimizer_state` maps their per-parameter state (Adam moments,
momentum buffers) and group hyperparameters onto the current groups, and
:func:`migrate_scheduler_state` widens the scheduler's per-group lists (``base_lrs``,
``_last_lr``, ...). When a saved state does not match the legacy layout of the current
model, the caller keeps the fresh optimizer and logs a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from training.training.lr_scaling import GROUP_ROLE_KEY, HEAD_ROLE

logger = logging.getLogger("lemtrain.training.param_groups")

BACKBONE_ROLE = "backbone"
HEAD_PREFIXES_ATTR = "head_param_prefixes"
"""Optional model attribute: parameter-name prefixes that make up the head."""

LEGACY_GROUP_COUNT = 2
"""Groups of a pre-split checkpoint: decay, then no-decay."""

PER_GROUP_SCHEDULER_KEYS = ("base_lrs", "_last_lr", "lr_lambdas", "min_lrs", "max_lrs")
"""Scheduler state lists holding one entry per param group."""


def _is_no_decay(name: str, param: nn.Parameter) -> bool:
    """Biases, norm and BatchNorm parameters and 1-D tensors get no weight decay."""
    lowered = name.lower()
    return param.ndim <= 1 or name.endswith(".bias") or "norm" in lowered or "bn" in lowered


def _trainable(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def head_prefixes(model: nn.Module) -> tuple[str, ...]:
    """Parameter-name prefixes of the model's head (empty when it has no separable head)."""
    declared = getattr(model, HEAD_PREFIXES_ATTR, None)
    if isinstance(declared, (list, tuple)) and all(isinstance(p, str) for p in declared):
        return tuple(declared)
    children = [
        name
        for name, child in model.named_children()
        if any(p.requires_grad for p in child.parameters())
    ]
    return (f"{children[-1]}.",) if len(children) >= 2 else ()


def _is_head(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == p.rstrip(".") or name.startswith(p) for p in prefixes)


def build_param_groups(model: nn.Module, base_wd: float) -> list[dict[str, Any]]:
    """Backbone/head x decay/no-decay groups, tagged with their role."""
    prefixes = head_prefixes(model)
    buckets: dict[tuple[str, bool], list[nn.Parameter]] = {
        (role, no_decay): [] for role in (BACKBONE_ROLE, HEAD_ROLE) for no_decay in (False, True)
    }
    for name, param in _trainable(model):
        role = HEAD_ROLE if _is_head(name, prefixes) else BACKBONE_ROLE
        buckets[(role, _is_no_decay(name, param))].append(param)
    groups: list[dict[str, Any]] = []
    for (role, no_decay), params in buckets.items():
        if role == HEAD_ROLE and not params:
            continue
        groups.append(
            {"params": params, "weight_decay": 0.0 if no_decay else base_wd, GROUP_ROLE_KEY: role}
        )
    return groups


def _legacy_order(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """The two pre-split groups of ``model``: decay params, then no-decay params."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in _trainable(model):
        (no_decay if _is_no_decay(name, param) else decay).append(param)
    return decay, no_decay


def _is_legacy_layout(saved: dict[str, Any], decay: int, no_decay: int) -> bool:
    groups = saved.get("param_groups")
    if not isinstance(groups, list) or len(groups) != LEGACY_GROUP_COUNT:
        return False
    if any(not isinstance(g, dict) or GROUP_ROLE_KEY in g for g in groups):
        return False
    sizes = [len(g.get("params", [])) for g in groups]
    return sizes == [decay, no_decay]


def legacy_group_sources(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[int]:
    """For each current param group, the legacy group (0 decay, 1 no-decay) it came from."""
    _decay, no_decay = _legacy_order(model)
    no_decay_ids = {id(p) for p in no_decay}
    sources: list[int] = []
    for group in optimizer.param_groups:
        params = group["params"]
        sources.append(1 if params and id(params[0]) in no_decay_ids else 0)
    return sources


def _remapped_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, saved: dict[str, Any]
) -> dict[int, Any]:
    """Per-parameter state of a legacy ``saved``, keyed by the current flat positions."""
    decay, no_decay = _legacy_order(model)
    legacy_ids = [int(i) for g in saved["param_groups"] for i in g["params"]]
    legacy_index = {id(p): legacy_ids[pos] for pos, p in enumerate([*decay, *no_decay])}
    saved_state: dict[Any, Any] = saved.get("state", {})
    flat = [p for g in optimizer.param_groups for p in g["params"]]
    state: dict[int, Any] = {}
    for position, param in enumerate(flat):
        old = legacy_index[id(param)]
        entry = saved_state.get(old, saved_state.get(str(old)))
        if entry is not None:
            state[position] = entry
    return state


def migrate_legacy_optimizer_state(
    model: nn.Module, optimizer: torch.optim.Optimizer, saved: dict[str, Any]
) -> dict[str, Any] | None:
    """``saved`` in the layout of ``optimizer``'s groups.

    A state whose group count already matches is returned unchanged. A legacy two-group
    state whose sizes match the current model's decay/no-decay split is remapped
    parameter by parameter; each group takes the hyperparameters (``lr``,
    ``initial_lr``, ``governor_lr_mult``, ...) of the legacy group it came from.
    Anything else returns ``None``.
    """
    groups = saved.get("param_groups")
    if isinstance(groups, list) and len(groups) == len(optimizer.param_groups):
        return saved
    decay, no_decay = _legacy_order(model)
    if not _is_legacy_layout(saved, len(decay), len(no_decay)):
        return None
    legacy_groups: list[dict[str, Any]] = saved["param_groups"]
    fresh_groups = optimizer.state_dict()["param_groups"]
    new_groups: list[dict[str, Any]] = []
    for fresh, source in zip(fresh_groups, legacy_group_sources(model, optimizer)):
        merged = {k: v for k, v in legacy_groups[source].items() if k != "params"}
        merged[GROUP_ROLE_KEY] = fresh.get(GROUP_ROLE_KEY)
        merged["params"] = list(fresh["params"])
        new_groups.append(merged)
    return {"state": _remapped_state(model, optimizer, saved), "param_groups": new_groups}


def migrate_scheduler_state(
    state: dict[str, Any], sources: Sequence[int], legacy_count: int = LEGACY_GROUP_COUNT
) -> dict[str, Any]:
    """Widen the scheduler's per-group lists from the legacy groups to ``sources``."""
    migrated = dict(state)
    for key in PER_GROUP_SCHEDULER_KEYS:
        value = state.get(key)
        if isinstance(value, list) and len(value) == legacy_count != len(sources):
            migrated[key] = [value[s] for s in sources]
    return migrated
