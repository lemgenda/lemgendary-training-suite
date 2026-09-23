from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from training.parallel.base import ParallelStrategy
from training.parallel.single import SingleGPUStrategy
from training.parallel.dp import DataParallelStrategy
from training.parallel.ddp import DistributedDataParallelStrategy

_STRATEGIES = {
    "single": SingleGPUStrategy,
    "dp": DataParallelStrategy,
    "ddp": DistributedDataParallelStrategy,
}


def available_strategies() -> list:
    return list(_STRATEGIES.keys())


def resolve_auto(model: Any, model_key: str, config: dict, model_info: dict | None = None) -> str:
    """Resolve the optimal parallel strategy based on hardware and model characteristics.

    Priority order:
        1. ``preferred_parallel`` key from ``model_info`` (explicit per-model override).
        2. Torchrun context detection via ``RANK`` env var — use ``ddp``.
        3. Single GPU or no CUDA — use ``single``.
        4. Multi-GPU default — use ``dp`` (safe single-process multi-GPU).

    Models that must always use ``single`` regardless of GPU count:
        - Forex: causal time-series, sequence variance breaks DP scatter/gather.
        - YOLO: Ultralytics native trainer handles its own DDP; external wrapping conflicts.
        - Small regressors (e.g. ``upn_v2``): DP overhead exceeds benefit.
    """
    info = model_info or {}

    # 1. Explicit per-model preference from the unified models registry.
    preferred = info.get("preferred_parallel", "")
    if preferred in _STRATEGIES:
        return preferred

    # 2. Models that must always run on a single process.
    _SINGLE_ONLY = {"forex", "yolo"}
    key_lower = model_key.lower()
    if any(tok in key_lower for tok in _SINGLE_ONLY):
        return "single"

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return "single"

    # 3. torchrun context: RANK is set, use DDP.
    if "RANK" in os.environ:
        return "ddp"

    # 4. Multi-GPU single-process fallback: use DP.
    return "dp"


def build_parallel_strategy(
    name: str,
    model: Any,
    device: Any,
    *,
    model_key: str = "",
    model_info: dict | None = None,
    config: dict | None = None,
) -> ParallelStrategy:
    config = config or {}
    if name == "auto":
        name = resolve_auto(model, model_key, config, model_info=model_info)
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown parallel strategy '{name}'. Available: {', '.join(available_strategies())}"
        )
    return _STRATEGIES[name](model, device, model_key=model_key, config=config)