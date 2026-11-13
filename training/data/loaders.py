"""Unified DataLoader factory functions and lifecycle management."""

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from training.data.workers import (
    UserWorkerCounts,
    compute_worker_topology,
    dataloader_worker_init_fn,
    dispose_loader,
)

logger = logging.getLogger("lemtrain.loaders")


@dataclass(frozen=True)
class LoaderPlacement:
    """Where a loader runs: worker topology is derived from these."""

    device: torch.device | None = None
    env: str = "local"
    config: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkerOverrides:
    """Worker settings that replace the hardware topology defaults; ``None`` keeps the default.

    Validation loaders never keep workers alive, so they ignore ``persistent_workers``.
    """

    num_workers: int | None = None
    persistent_workers: bool | None = None
    pin_memory: bool | None = None
    prefetch_factor: int | None = None


@dataclass(frozen=True)
class TrainLoaderOptions:
    """Training loader settings."""

    sampler: Sampler[Any] | None = None
    shuffle: bool = True
    drop_last: bool = True
    collate_fn: Any | None = None
    workers: WorkerOverrides = WorkerOverrides()


@dataclass(frozen=True)
class ValLoaderOptions:
    """Validation loader settings."""

    is_constrained: bool = False
    collate_fn: Any | None = None
    workers: WorkerOverrides = WorkerOverrides()


def _positive_batch_size(batch_size: Any, fallback: int) -> int:
    """``batch_size`` as a positive int, or ``fallback`` when it cannot be one."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            batch_size = fallback
    return batch_size


def _placement_device(placement: LoaderPlacement) -> torch.device:
    return placement.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_loader(
    dataset: Dataset[Any],
    batch_size: int,
    options: TrainLoaderOptions | None = None,
    placement: LoaderPlacement | None = None,
) -> DataLoader[Any]:
    """Construct a standardized training DataLoader with hardware-tuned worker topology."""
    opts = options or TrainLoaderOptions()
    placement = placement or LoaderPlacement()
    dev = _placement_device(placement)
    is_forex = getattr(dataset, "task_type", "") == "forex"
    batch_size = _positive_batch_size(batch_size, 128 if is_forex else 16)

    topology = compute_worker_topology(
        env=placement.env,
        device=dev,
        is_forex=is_forex,
        user_workers=UserWorkerCounts(train=opts.workers.num_workers),
        config=placement.config,
    )

    over = opts.workers
    workers = topology.num_workers if over.num_workers is None else over.num_workers
    pin = topology.pin_memory if over.pin_memory is None else over.pin_memory
    persistent = (
        topology.persistent_workers if over.persistent_workers is None else over.persistent_workers
    )
    if workers == 0:
        persistent = False

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "persistent_workers": persistent,
        "pin_memory": pin,
        "drop_last": opts.drop_last,
    }

    if workers > 0:
        kwargs["worker_init_fn"] = dataloader_worker_init_fn

    if opts.collate_fn is not None:
        kwargs["collate_fn"] = opts.collate_fn

    if opts.sampler is not None:
        kwargs["sampler"] = opts.sampler
    # Automatically use DistributedSampler when running multi-GPU DDP
    elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        kwargs["sampler"] = DistributedSampler(dataset, shuffle=opts.shuffle)
    else:
        kwargs["shuffle"] = opts.shuffle

    if workers > 0:
        kwargs["prefetch_factor"] = over.prefetch_factor or topology.prefetch_factor or 4

    logger.debug(
        "Building training DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s, persistent=%s",
        batch_size,
        workers,
        pin,
        persistent,
    )
    return DataLoader(dataset, **kwargs)


def build_val_loader(
    dataset: Dataset[Any],
    batch_size: int,
    options: ValLoaderOptions | None = None,
    placement: LoaderPlacement | None = None,
) -> DataLoader[Any]:
    """Construct a standardized validation DataLoader with conservative worker overhead."""
    opts = options or ValLoaderOptions()
    placement = placement or LoaderPlacement()
    dev = _placement_device(placement)
    is_forex = getattr(dataset, "task_type", "") == "forex"
    batch_size = _positive_batch_size(batch_size, 256 if is_forex else 16)

    topology = compute_worker_topology(
        env=placement.env,
        device=dev,
        is_forex=is_forex,
        user_workers=UserWorkerCounts(val=opts.workers.num_workers),
        config=placement.config,
    )

    over = opts.workers
    workers = topology.val_num_workers if over.num_workers is None else over.num_workers
    pin = topology.pin_memory if over.pin_memory is None else over.pin_memory

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": False,
        "pin_memory": pin,
        "drop_last": False,
    }

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        kwargs["sampler"] = DistributedSampler(dataset, shuffle=False)

    if workers > 0:
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
        kwargs["prefetch_factor"] = _val_prefetch_factor(dataset, opts)
    if opts.collate_fn is not None:
        kwargs["collate_fn"] = opts.collate_fn

    logger.info(
        "Building validation DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s",
        batch_size,
        workers,
        pin,
    )
    return DataLoader(dataset, **kwargs)


def _val_prefetch_factor(dataset: Dataset[Any], opts: ValLoaderOptions) -> int:
    """Explicit prefetch, else 4 for constrained runs or >=512px validation, else 6."""
    val_res = getattr(dataset, "size", (256, 256))
    val_h = val_res[0] if isinstance(val_res, (list, tuple)) else val_res
    if opts.workers.prefetch_factor is not None:
        return opts.workers.prefetch_factor
    if opts.is_constrained or (isinstance(val_h, int) and val_h >= 512):
        return 4
    return 6


def rebuild_train_loader(
    old_loader: DataLoader[Any] | None,
    dataset: Dataset[Any],
    batch_size: int,
    options: TrainLoaderOptions | None = None,
    placement: LoaderPlacement | None = None,
) -> DataLoader[Any]:
    """Safely dispose of previous DataLoader worker processes and return a fresh DataLoader.

    Without an explicit ``collate_fn`` the old loader's collate function is kept.
    """
    opts = options or TrainLoaderOptions()
    if old_loader is not None:
        if opts.collate_fn is None:
            opts = dataclasses.replace(opts, collate_fn=getattr(old_loader, "collate_fn", None))
        dispose_loader(old_loader)

    return build_train_loader(
        dataset=dataset,
        batch_size=batch_size,
        options=opts,
        placement=placement,
    )
