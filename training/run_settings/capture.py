"""Capture the frozen and epoch settings of a run from its config, registry entry and CLI.

Each value comes from the reader its consumer uses (SSOT): ``LocalityConfig`` for the
shard locality, the governor setup for the effective batch, ``optimizer_settings`` for the
optimizer, LR, weight decay and gradient clip, ``risk_per_trade`` and
``registry_emd_form`` for the metric definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from training.config.optimizer_settings import (
    base_learning_rate,
    base_weight_decay,
    grad_clip_norm,
    optimizer_shape,
)
from training.context_data import requested_batch_size
from training.data.locality import FRACTION_SEED, LocalityConfig
from training.governance.governor_setup import (
    configured_accumulation,
    configured_target_effective_batch,
)
from training.governance.quality_score import FOREX_TASKS, quality_scheme
from training.loss_emd import registry_emd_form
from training.registry.forex_risk import risk_per_trade
from training.run_settings.keys import (
    BATCH,
    GRAD_CLIP,
    HARDWARE_SECTION,
    LOADER,
    LOADER_ARGS,
    LOCALITY,
    LR_SCHEDULE,
    WEIGHT_DECAY,
)

QUALITY_TASK = "quality"
"""Task type whose loss has an EMD form (NIMA)."""

PRECISION_KEYS: tuple[str, ...] = ("use_fp16", "amp_enabled", "use_bf16")


@dataclass(frozen=True)
class SettingsInputs:
    """What a run's settings are read from.

    ``param_groups`` is the optimizer's group layout as ``(role, parameter count)`` pairs;
    ``total_epochs`` is the configured epoch budget the LR schedule is built for.
    """

    config: Mapping[str, Any]
    model_info: Mapping[str, Any]
    args: Any
    task_type: str
    total_epochs: int
    model_class: str
    param_groups: tuple[tuple[str, int], ...] = ()


def param_group_layout(param_groups: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, int], ...]:
    """``(role, parameter count)`` of each optimizer param group."""
    return tuple(
        (str(group.get("group", "")), len(group.get("params", ()))) for group in param_groups
    )


def _hardware(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config.get(HARDWARE_SECTION)
    return section if isinstance(section, Mapping) else {}


def frozen_settings(inputs: SettingsInputs) -> dict[str, Any]:
    """Settings fixed for the whole run; a resume refuses a change (``--override-frozen``)."""
    task_type = inputs.task_type
    info = inputs.model_info
    risk = risk_per_trade(info) if task_type in FOREX_TASKS else None
    hardware = _hardware(inputs.config)
    return {
        "model.class": inputs.model_class,
        "model.kwargs": info.get("kwargs") or {},
        "model.param_groups": [list(group) for group in inputs.param_groups],
        "optimizer": optimizer_shape(inputs.config),
        # The compiled datasets fix the train/validation split. The walk-forward fold and
        # the CLI pairs/timeframe stage are curriculum choices a run moves through.
        "data.datasets": {
            "datasets": info.get("datasets") or [],
            "dataset_type": info.get("dataset_type"),
            "task_type": task_type,
        },
        "metrics.scheme": quality_scheme(task_type, risk)
        if risk is not None
        else quality_scheme(task_type),
        "evaluation.risk_per_trade": risk,
        "stabilizers.emd_form": (
            registry_emd_form(info.get("stabilizers")) if task_type == QUALITY_TASK else None
        ),
        "seed": {"fraction_seed": FRACTION_SEED},
        "precision": {key: hardware.get(key) for key in PRECISION_KEYS},
    }


def epoch_settings(inputs: SettingsInputs) -> dict[str, Any]:
    """Settings that take effect at an epoch boundary (see :data:`EPOCH_SETTING_KEYS`)."""
    config, info, args = inputs.config, dict(inputs.model_info), inputs.args
    hardware = _hardware(config)
    loader = {name: getattr(args, name, None) for name in LOADER_ARGS}
    loader.update({f"{HARDWARE_SECTION}.{name}": hardware.get(name) for name in LOADER_ARGS})
    return {
        LOCALITY: LocalityConfig.from_config(config).as_dict(),
        BATCH: {
            "batch_size": requested_batch_size(args, info),
            "target_effective_batch": configured_target_effective_batch(info, dict(config)),
            "accumulation_steps": configured_accumulation(info),
        },
        LR_SCHEDULE: {
            "scheduler": dict(config.get("scheduler") or {}),
            "lr": base_learning_rate(config, getattr(args, "lr", None)),
            "total_epochs": int(inputs.total_epochs),
        },
        WEIGHT_DECAY: base_weight_decay(config),
        GRAD_CLIP: grad_clip_norm(config),
        LOADER: loader,
    }
