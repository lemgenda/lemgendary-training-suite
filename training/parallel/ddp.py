from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any
import torch
import torch.nn as nn
from training.parallel.base import ParallelStrategy, generic_prepare_batch

logger = logging.getLogger("lemtrain.parallel.ddp")


class DistributedDataParallelStrategy(ParallelStrategy):
    """Full DistributedDataParallel strategy compatible with torchrun launches.

    Process group is initialized from standard torchrun environment variables:
        RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK.

    For single-machine multi-GPU (the common Kaggle/cloud case), torchrun sets
    all of these automatically. For CPU-only environments, the Gloo backend is
    used as a fallback.
    """

    name = "ddp"

    def __init__(self, model, device, *, model_key="", config=None):
        super().__init__(model, device, model_key=model_key, config=config)
        self._rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._initialized = False

    def setup(self) -> nn.Module:
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", "29500")
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", master_port)

            dist.init_process_group(
                backend=backend,
                rank=self._rank,
                world_size=self._world_size,
            )
            self._initialized = True
            logger.info(
                "DDP process group initialized: backend=%s rank=%d/%d",
                backend,
                self._rank,
                self._world_size,
            )

        # Move model to the correct GPU for this rank before wrapping.
        if torch.cuda.is_available():
            local_device = torch.device(f"cuda:{self._local_rank}")
            self.raw_model.to(local_device)
            self._wrapped = nn.parallel.DistributedDataParallel(
                self.raw_model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=False,
            )
        else:
            # CPU fallback (Gloo backend).
            self.raw_model.to(self.device)
            self._wrapped = nn.parallel.DistributedDataParallel(
                self.raw_model,
                find_unused_parameters=False,
            )

        logger.info(
            "DDP model wrapped: rank=%d local_rank=%d world_size=%d",
            self._rank,
            self._local_rank,
            self._world_size,
        )
        return self._wrapped

    def cleanup(self) -> None:
        """Tear down the distributed process group on exit."""
        if self._initialized:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.debug("DDP process group destroyed for rank %d.", self._rank)
            self._initialized = False

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    def prepare_batch(self, inputs: Any, targets: Any, tasks: Any) -> tuple[Any, Any, Any]:
        local_device = (
            torch.device(f"cuda:{self._local_rank}")
            if torch.cuda.is_available()
            else self.device
        )
        inputs, targets, _ = generic_prepare_batch(
            inputs, targets, tasks, local_device, self._task_type()
        )
        if isinstance(tasks, torch.Tensor):
            tasks = tasks.to(local_device, non_blocking=True)
        return inputs, targets, tasks

    def backward_context(self, accumulation_steps: int) -> AbstractContextManager[Any]:
        """Suppress all-reduce on non-final gradient accumulation micro-batches."""
        if accumulation_steps > 1 and self._wrapped is not None and hasattr(self._wrapped, "no_sync"):
            return self._wrapped.no_sync()
        return nullcontext()

    def state_dict_for_save(self) -> dict:
        """Return raw (unwrapped) model state dict for checkpointing."""
        return self.normalize_state_dict(self.raw_model.state_dict())