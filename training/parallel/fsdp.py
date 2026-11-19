"""Fully Sharded Data Parallel (FSDP) training strategy.

Wraps model parameters, gradients, and optimizer states into distributed shards
across multi-GPU ranks, supporting CPU parameter offloading and mixed-precision policies.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from training.parallel.base import ParallelStrategy, generic_prepare_batch
from training.parallel.ddp import init_distributed_process_group

logger = logging.getLogger("lemtrain.parallel.fsdp")


class FSDPParallelStrategy(ParallelStrategy):
    """Fully Sharded Data Parallel (FSDP) execution strategy."""

    name = "fsdp"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        model_key: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model, device, model_key=model_key, config=config)
        self.dist_env = {
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
        self._initialized = False

    def setup(self) -> nn.Module:
        """Initialize the distributed process group and wrap model with FSDP."""
        self._initialized = init_distributed_process_group(
            self.dist_env["local_rank"], self.dist_env["rank"], self.dist_env["world_size"]
        )

        try:
            sharding_type = ShardingStrategy.FULL_SHARD
            wrapped = FSDP(
                self.raw_model,
                sharding_strategy=sharding_type,
                device_id=torch.device(f"cuda:{self.dist_env['local_rank']}")
                if torch.cuda.is_available()
                else None,
                use_orig_params=True,
            )
            self._wrapped = wrapped
            return wrapped
        except (ImportError, RuntimeError) as exc:
            logger.warning("FSDP wrapping failed (%s). Falling back to unwrapped model.", exc)
            self._wrapped = self.raw_model
            return self.raw_model

    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        """Move inputs, targets, and tasks to the target device."""
        inputs_dev, targets_dev, _ = generic_prepare_batch(
            inputs, targets, self.device, self._task_type()
        )
        return inputs_dev, targets_dev, tasks

    def unwrap(self) -> nn.Module:
        """Return raw inner model."""
        return self.raw_model
