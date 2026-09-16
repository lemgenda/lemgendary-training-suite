from __future__ import annotations
from contextlib import AbstractContextManager, nullcontext
from typing import Any
import torch
import torch.nn as nn
from training.parallel.base import ParallelStrategy, generic_prepare_batch


class DistributedDataParallelStrategy(ParallelStrategy):
    name = "ddp"

    def __init__(self, model, device, *, model_key="", config=None):
        super().__init__(model, device, model_key=model_key, config=config)
        self._rank = 0
        self._world_size = 1
        self._initialized = False

    def setup(self) -> nn.Module:
        raise NotImplementedError(
            "DistributedDataParallel is scaffolded but not yet wired into core_loop.py. "
            "See training_refactoring.md Phase 10.5. Use --parallel single or --parallel dp."
        )

    def cleanup(self) -> None:
        if self._initialized:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            self._initialized = False

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        local_device = torch.device(f"cuda:{self._rank}")
        inputs, targets, _ = generic_prepare_batch(
            inputs, targets, tasks, local_device, self._task_type()
        )
        if isinstance(tasks, torch.Tensor):
            tasks = tasks.to(local_device, non_blocking=True)
        return inputs, targets, tasks

    def backward_context(self, accumulation_steps: int) -> AbstractContextManager[Any]:
        # DDP suppresses all-reduce on non-final micro-batches via .no_sync().
        # We return the context manager directly instead of using @contextmanager so
        # the return type matches the base class exactly.
        if accumulation_steps > 1 and self._wrapped is not None and hasattr(self._wrapped, "no_sync"):
            return self._wrapped.no_sync()
        return nullcontext()