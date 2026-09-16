from __future__ import annotations
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


def resolve_auto(model, model_key, config):
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return "single"
    if "forex" in model_key.lower():
        return "single"
    return "dp"


def build_parallel_strategy(name, model, device, *, model_key="", config=None):
    config = config or {}
    if name == "auto":
        name = resolve_auto(model, model_key, config)
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown parallel strategy '{name}'. Available: {', '.join(available_strategies())}"
        )
    return _STRATEGIES[name](model, device, model_key=model_key, config=config)