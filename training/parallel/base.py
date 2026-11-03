"""Abstract parallel training strategy and shared batch-to-device helper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn


class ParallelStrategy(ABC):
    """Base class wrapping a model for single/multi-device training and checkpoint I/O."""

    name = "base"

    def __init__(self, model, device, *, model_key="", config=None):
        self.raw_model = model
        self.device = device
        self.model_key = model_key
        self.config = config or {}
        self._wrapped = None

    @abstractmethod
    def setup(self) -> nn.Module:
        """Wrap and place ``raw_model`` on its device(s); return the wrapped module."""

    def cleanup(self) -> None:
        """Release strategy resources (process groups etc.); no-op by default."""

    @property
    def model(self) -> nn.Module:
        """The wrapped model; raises if :meth:`setup` has not run."""
        if self._wrapped is None:
            raise RuntimeError(f"{type(self).__name__}.setup() not called yet.")
        return self._wrapped

    @property
    def world_size(self) -> int:
        """Number of participating processes (1 for single-process strategies)."""
        return 1

    @property
    def rank(self) -> int:
        """Global rank of this process (0 for single-process strategies)."""
        return 0

    @property
    def is_main_process(self) -> bool:
        """True on the rank responsible for logging and checkpointing."""
        return self.rank == 0

    def _task_type(self) -> str:
        return self.config.get("task_type", "")

    @abstractmethod
    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        """Move a batch to this strategy's device(s); return ``(inputs, targets, tasks)``."""

    def backward_context(self) -> AbstractContextManager[Any]:
        """Context for a non-final gradient-accumulation backward (DDP skips all-reduce)."""
        return nullcontext()

    def state_dict_for_save(self) -> dict:
        """Return the unwrapped model state dict for checkpointing."""
        return self.normalize_state_dict(self.raw_model.state_dict())

    def normalize_state_dict(self, state_dict: dict) -> dict:
        """Strip a leading ``module.`` wrapper prefix from state-dict keys."""
        if not state_dict:
            return state_dict
        has_prefix = any(k.startswith("module.") for k in state_dict)
        if not has_prefix:
            return state_dict
        return {
            (k.replace("module.", "", 1) if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        """Load a (possibly ``module.``-prefixed) state dict into ``raw_model``."""
        clean = self.normalize_state_dict(state_dict)
        self.raw_model.load_state_dict(clean, strict=strict)


def generic_prepare_batch(inputs, targets, device, task_type) -> tuple[Any, Any, Any]:
    """Move inputs/targets (tensors or dicts of tensors) to ``device`` per task type.

    Returns ``(inputs, targets, None)``; callers move task indices themselves.
    """
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
