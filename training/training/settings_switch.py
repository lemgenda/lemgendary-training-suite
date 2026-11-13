"""Apply a due epoch-settings change set at an epoch boundary (``training.run_settings``).

Every epoch setting has an applier (a registry, so a new setting adds an entry and the
epoch loop stays unchanged). An applier returns False when it refuses a change; the
recorded value then stays active and the refusal is logged and recorded:

* ``data.locality``, ``grad_clip``, ``loader``: written into the run's config and CLI
  namespace (``pinning``); the train loader is rebuilt for locality and loader changes;
* ``batch``: accepted only when the accumulation ``effective / batch`` is an exact
  integer; the governor curriculum then takes the new batch and accumulation;
* ``lr_schedule``: the scheduler is rebuilt from the new settings and fast-forwarded to
  the number of scheduler steps already taken (the timm resume approach); the old
  scheduler state is never combined with the new parameters;
* ``weight_decay``: set on the decaying param groups.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.run_settings.keys import (
    BATCH,
    GRAD_CLIP,
    LOADER,
    LOCALITY,
    LR_SCHEDULE,
    WEIGHT_DECAY,
)
from training.run_settings.ledger import report_changes
from training.run_settings.pinning import pin_loader, pin_locality, pin_optimizer_value
from training.run_settings.record import SettingChange
from training.training.context import TrainingContext
from training.training.lr_scaling import APPLIED_MULT_KEY
from training.training.optimizer import build_scheduler

logger = logging.getLogger("lemtrain.run_settings")

Applier = Callable[[TrainingContext, SettingChange], bool]
LOADER_REBUILD_KEYS = frozenset({LOCALITY, BATCH, LOADER})
"""Settings whose change needs a new train loader."""


def scheduler_steps(scheduler: Any) -> int:
    """Scheduler steps taken so far (``last_epoch``; 0 without a scheduler)."""
    return max(0, int(getattr(scheduler, "last_epoch", 0) or 0))


def _build(optimizer: torch.optim.Optimizer, schedule: Mapping[str, Any]) -> Any:
    return build_scheduler(
        optimizer=optimizer,
        config={"scheduler": dict(schedule.get("scheduler") or {})},
        total_epochs=int(schedule["total_epochs"]),
    )


def build_recorded_scheduler(optimizer: torch.optim.Optimizer, schedule: Mapping[str, Any]) -> Any:
    """A scheduler built from a recorded ``lr_schedule`` group (to load its state into)."""
    return _build(optimizer, schedule)


def rebuild_scheduler(
    optimizer: torch.optim.Optimizer, steps: int, schedule: Mapping[str, Any]
) -> Any:
    """A scheduler built from ``schedule`` and fast-forwarded by ``steps`` steps.

    Every group restarts from ``schedule['lr']`` times the governor multiplier it carries.
    The fast-forward runs on a scratch optimizer with the same groups, and the real
    scheduler loads its state, so the real optimizer is never stepped.
    """
    base_lr = float(schedule["lr"])
    for group in optimizer.param_groups:
        scaled = base_lr * float(group.get(APPLIED_MULT_KEY, 1.0))
        group["lr"] = scaled
        group["initial_lr"] = scaled
    scratch = torch.optim.SGD(
        [
            {"params": [torch.zeros(1, requires_grad=True)], "lr": float(group["lr"])}
            for group in optimizer.param_groups
        ]
    )
    for scratch_group, group in zip(scratch.param_groups, optimizer.param_groups, strict=True):
        scratch_group["initial_lr"] = group["initial_lr"]
    replay = _build(scratch, schedule)
    if not isinstance(replay, ReduceLROnPlateau):
        for _ in range(max(0, steps)):
            scratch.step()
            replay.step()
    scheduler = _build(optimizer, schedule)
    scheduler.load_state_dict(replay.state_dict())
    for group, scratch_group in zip(optimizer.param_groups, scratch.param_groups, strict=True):
        group["lr"] = scratch_group["lr"]
    return scheduler


def apply_lr_schedule(ctx: TrainingContext, change: SettingChange) -> bool:
    """Rebuild the scheduler from the new schedule at the steps already taken."""
    steps = scheduler_steps(ctx.models.scheduler)
    ctx.models.scheduler = rebuild_scheduler(ctx.models.optimizer, steps, change.new)
    logger.info(
        "[SETTINGS] LR scheduler rebuilt from %s and fast-forwarded %d steps (LR %.6g).",
        change.new,
        steps,
        float(ctx.models.optimizer.param_groups[0]["lr"]),
    )
    return True


def apply_batch(ctx: TrainingContext, change: SettingChange) -> bool:
    """Take the new batch when ``effective / batch`` is an exact accumulation.

    The effective batch is ``batch * accumulation_steps`` when an accumulation is
    configured, else ``target_effective_batch``; an ``auto`` batch keeps the current one.
    """
    curriculum = getattr(ctx.governance.governor, "curriculum", None)
    if curriculum is None:
        logger.warning("[SETTINGS] The governor has no curriculum; batch change refused.")
        return False
    new = change.new
    batch = int(new.get("batch_size") or curriculum.current_batch)
    configured_acc = new.get("accumulation_steps")
    effective = (
        batch * int(configured_acc) if configured_acc else int(new["target_effective_batch"])
    )
    if batch <= 0 or effective % batch != 0:
        logger.warning(
            "[SETTINGS] Batch %d does not divide the effective batch %d (accumulation must "
            "be an integer); keeping batch %d x accumulation %d.",
            batch,
            effective,
            curriculum.current_batch,
            curriculum.current_acc,
        )
        return False
    curriculum.target_effective_batch = effective
    curriculum.current_batch = batch
    curriculum.current_acc = effective // batch
    ctx.data.accumulation_steps = curriculum.current_acc
    return True


def apply_weight_decay(ctx: TrainingContext, change: SettingChange) -> bool:
    """Set the new weight decay on the groups that decayed with the old one."""
    old, new = float(change.old or 0.0), float(change.new)
    if old <= 0.0:
        logger.warning(
            "[SETTINGS] The run trained without weight decay, so its decaying groups are "
            "unknown; weight decay %.6g refused.",
            new,
        )
        return False
    for group in ctx.models.optimizer.param_groups:
        if float(group.get("weight_decay", 0.0)) > 0.0:
            group["weight_decay"] = new
    pin_optimizer_value(ctx.run.config, "weight_decay", new)
    return True


def apply_locality(ctx: TrainingContext, change: SettingChange) -> bool:
    """Write the new ``data.locality`` into the config (the loader rebuild reads it)."""
    pin_locality(ctx.run.config, change.new)
    return True


def apply_grad_clip(ctx: TrainingContext, change: SettingChange) -> bool:
    """Write the new ``optimizer.grad_clip`` (the next train pass reads it)."""
    pin_optimizer_value(ctx.run.config, "grad_clip", change.new)
    return True


def apply_loader(ctx: TrainingContext, change: SettingChange) -> bool:
    """Write the new worker counts (the loader rebuild reads them)."""
    pin_loader(ctx.run.config, ctx.run.args, change.new)
    return True


EPOCH_APPLIERS: dict[str, Applier] = {
    LOCALITY: apply_locality,
    BATCH: apply_batch,
    LR_SCHEDULE: apply_lr_schedule,
    WEIGHT_DECAY: apply_weight_decay,
    GRAD_CLIP: apply_grad_clip,
    LOADER: apply_loader,
}


def apply_due_settings(
    ctx: TrainingContext, epoch: int, rebuild_loader: Callable[[TrainingContext], None]
) -> None:
    """Apply the change set due when epoch ``epoch`` starts, then log and record it."""
    tracker = ctx.state.run_settings
    if tracker is None:
        return
    due = tracker.due(epoch)
    if due is None or not due.changes:
        return
    applied: dict[str, SettingChange] = {}
    refused: dict[str, SettingChange] = {}
    for key, change in due.changes.items():
        applier = EPOCH_APPLIERS.get(key)
        if applier is None:
            logger.warning("[SETTINGS] No applier for setting %r; recorded as changed.", key)
            applied[key] = change
        elif applier(ctx, change):
            applied[key] = change
        else:
            refused[key] = change
    tracker.activate({key: change.new for key, change in applied.items()})
    if LOADER_REBUILD_KEYS & applied.keys():
        rebuild_loader(ctx)
    report_changes(tracker.ledger, epoch, due.kind, applied, refused)
