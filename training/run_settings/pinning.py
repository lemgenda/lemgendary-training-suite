"""Write epoch settings back into the run's config and CLI namespace.

The loaders, the train step and the subset builder read ``data.locality``,
``optimizer.grad_clip``, ``optimizer.weight_decay`` and the worker counts from the
run's ``config`` and ``args``. Writing a settings group there makes every later reader
see it: the recorded group while an interrupted epoch finishes, the configured group
from the next epoch on. Batch size and the LR schedule live in the governor and the
scheduler instead (``training.training.settings_switch``). This module is torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.config.optimizer_settings import OPTIMIZER_SECTION
from training.data.locality import LocalityConfig
from training.run_settings.keys import (
    GRAD_CLIP,
    HARDWARE_SECTION,
    LOADER,
    LOADER_ARGS,
    LOCALITY,
    WEIGHT_DECAY,
)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """``config[name]`` as a dict, created (or replaced when not a dict) in place."""
    section = config.get(name)
    if not isinstance(section, dict):
        section = {}
        config[name] = section
    return section


def pin_locality(config: dict[str, Any], value: Mapping[str, Any]) -> None:
    """Write a ``data.locality`` settings group into ``config``."""
    LocalityConfig.from_section(value).pin(config)


def pin_optimizer_value(config: dict[str, Any], key: str, value: Any) -> None:
    """Write ``optimizer.<key>`` into ``config``."""
    _section(config, OPTIMIZER_SECTION)[key] = value


def pin_loader(config: dict[str, Any], args: Any, value: Mapping[str, Any]) -> None:
    """Write the worker counts into ``args`` and ``config['hardware']``."""
    hardware = _section(config, HARDWARE_SECTION)
    for name in LOADER_ARGS:
        if args is not None:
            setattr(args, name, value.get(name))
        configured = value.get(f"{HARDWARE_SECTION}.{name}")
        if configured is None:
            hardware.pop(name, None)
        else:
            hardware[name] = configured


def pin_config_settings(config: dict[str, Any], args: Any, settings: Mapping[str, Any]) -> None:
    """Write every config- or CLI-held setting of ``settings`` into ``config`` and ``args``."""
    if LOCALITY in settings:
        pin_locality(config, settings[LOCALITY])
    if GRAD_CLIP in settings:
        pin_optimizer_value(config, "grad_clip", settings[GRAD_CLIP])
    if WEIGHT_DECAY in settings:
        pin_optimizer_value(config, "weight_decay", settings[WEIGHT_DECAY])
    if LOADER in settings:
        pin_loader(config, args, settings[LOADER])
