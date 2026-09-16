from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any
import torch
import torch.nn as nn


class ParallelStrategy(ABC):
    name = "base"

    def __init__(self, model, device, *, model_key="", config=None):
        self.raw_model = model
        self.device = device
        self.model_key = model_key
        self.config = config or {}
        self._wrapped = None

    @abstractmethod
    def setup(self) -> nn.Module:
        ...

    def cleanup(self) -> None:
        return None

    @property
    def model(self) -> nn.Module:
        if self._wrapped is None:
            raise RuntimeError(f"{type(self).__name__}.setup() not called yet.")
        return self._wrapped

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def _task_type(self) -> str:
        return self.config.get("task_type", "")

    @abstractmethod
    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        ...

    def backward_context(self, accumulation_steps: int) -> AbstractContextManager[Any]:
        return nullcontext()

    def state_dict_for_save(self) -> dict:
        return self.normalize_state_dict(self.raw_model.state_dict())

    def normalize_state_dict(self, state_dict: dict) -> dict:
        if not state_dict:
            return state_dict
        has_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if not has_prefix:
            return state_dict
        return {
            (k.replace("module.", "", 1) if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        clean = self.normalize_state_dict(state_dict)
        self.raw_model.load_state_dict(clean, strict=strict)


def generic_prepare_batch(inputs, targets, tasks, device, task_type) -> tuple[Any, Any, Any]:
    if task_type in ("text_to_image", "image_to_text"):
        if isinstance(inputs, dict):
            inputs = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
        return inputs, None, None

    if task_type == "forex":
        if isinstance(inputs, dict):
            inputs = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device, non_blocking=True)
        if isinstance(targets, dict):
            targets = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in targets.items()
            }
        elif isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=True)
        return inputs, targets, None

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device, non_blocking=True)
    elif isinstance(inputs, dict):
        inputs = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
    if isinstance(targets, dict):
        targets = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in targets.items()
        }
    elif isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    return inputs, targets, None