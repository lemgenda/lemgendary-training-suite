"""Unified DataLoader factory functions and lifecycle management."""

import logging
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from training.data.workers import compute_worker_topology, dispose_loader

logger = logging.getLogger("lemtrain.loaders")


def build_train_loader(
    dataset: Dataset[Any],
    batch_size: int,
    sampler: Sampler[Any] | None = None,
    shuffle: bool = True,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    drop_last: bool = True,
    device: torch.device | None = None,
    env: str = "local",
    config: dict[str, Any] | None = None,
) -> DataLoader[Any]:
    """Construct a standardized training DataLoader with hardware-tuned worker topology."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_forex = getattr(dataset, "task_type", "") == "forex"

    topology = compute_worker_topology(
        env=env,
        device=dev,
        is_forex=is_forex,
        user_num_workers=num_workers,
        config=config,
    )

    workers = topology.num_workers if num_workers is None else num_workers
    pin = topology.pin_memory if pin_memory is None else pin_memory
    persistent = topology.persistent_workers if persistent_workers is None else persistent_workers
    if workers == 0:
        persistent = False

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "persistent_workers": persistent,
        "pin_memory": pin,
        "drop_last": drop_last,
    }

    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle

    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor or topology.prefetch_factor or 4

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
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    device: torch.device | None = None,
    env: str = "local",
    is_constrained: bool = False,
    config: dict[str, Any] | None = None,
) -> DataLoader[Any]:
    """Construct a standardized validation DataLoader with conservative worker overhead."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_forex = getattr(dataset, "task_type", "") == "forex"

    topology = compute_worker_topology(
        env=env,
        device=dev,
        is_forex=is_forex,
        user_val_workers=num_workers,
        config=config,
    )

    workers = topology.val_num_workers if num_workers is None else num_workers
    pin = topology.pin_memory if pin_memory is None else pin_memory

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": False,
        "pin_memory": pin,
        "drop_last": False,
    }

    if workers > 0:
        val_res = getattr(dataset, "size", (256, 256))
        val_h = val_res[0] if isinstance(val_res, (list, tuple)) else val_res
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        elif is_constrained or (isinstance(val_h, int) and val_h >= 512):
            kwargs["prefetch_factor"] = 4
        else:
            kwargs["prefetch_factor"] = 6

    logger.debug(
        "Building validation DataLoader: batch_size=%d, num_workers=%d, pin_memory=%s",
        batch_size,
        workers,
        pin,
    )
    return DataLoader(dataset, **kwargs)


def rebuild_train_loader(
    old_loader: DataLoader[Any] | None,
    dataset: Dataset[Any],
    batch_size: int,
    sampler: Sampler[Any] | None = None,
    shuffle: bool = True,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    drop_last: bool = True,
    device: torch.device | None = None,
    env: str = "local",
    config: dict[str, Any] | None = None,
) -> DataLoader[Any]:
    """Safely dispose of previous DataLoader worker processes and return a fresh DataLoader."""
    if old_loader is not None:
        dispose_loader(old_loader)

    return build_train_loader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        device=device,
        env=env,
        config=config,
    )
