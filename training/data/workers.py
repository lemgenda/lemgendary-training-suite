"""Worker topology calculation and memory-leak-free DataLoader disposal."""

import gc
import logging
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any

import psutil
import torch

logger = logging.getLogger("lemtrain.workers")


@dataclass(frozen=True)
class WorkerTopology:
    """Calculated execution topology for PyTorch DataLoaders."""

    num_workers: int
    val_num_workers: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int | None
    num_threads: int


@dataclass(frozen=True)
class UserWorkerCounts:
    """User-requested train/validation worker counts; ``None`` keeps the computed default."""

    train: int | None = None
    val: int | None = None


def _system_ram_gb() -> float:
    """Total system RAM in GB (16.0 when psutil cannot tell)."""
    try:
        return psutil.virtual_memory().total / (1024**3)
    except OSError as exc:
        logger.debug("psutil unavailable for worker topology calculation: %s", exc)
        return 16.0


def _set_cpu_threads(env: str, cpu_count: int) -> int:
    """Set torch CPU threads (1 on Kaggle, else every core); return the count."""
    num_threads = 1 if env == "kaggle" else max(1, cpu_count)
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError as exc:
        logger.debug("Failed setting torch CPU threads to %d: %s", num_threads, exc)
    return num_threads


def _default_train_workers(env: str, ram_gb: float, cpu_count: int, cfg: dict[str, Any]) -> int:
    """Training DataLoader workers for the environment and platform."""
    if env in {"kaggle", "colab"}:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # In multi-GPU DDP on Kaggle (2 GPUs), 4 workers per process spawns 8 total workers,
        # exhausting shared memory and RAM. Cap to 2 workers per process in DDP.
        return 2 if world_size > 1 else 4
    if sys.platform == "win32":
        # Windows multi-process PyTorch spawns via freeze_support; restrict to 2 if ample RAM
        return 2 if ram_gb >= 16.0 else 0
    cfg_workers = cfg.get("hardware", {}).get("num_workers", 4)
    if isinstance(cfg_workers, int):
        return min(cpu_count, cfg_workers)
    return min(cpu_count, 4)


def _default_val_workers(env: str, num_workers: int, cfg: dict[str, Any]) -> int:
    """Validation workers: ``hardware.val_num_workers`` if an int, else per environment."""
    cfg_val = cfg.get("hardware", {}).get("val_num_workers", "auto")
    if isinstance(cfg_val, int):
        return cfg_val
    if env in {"kaggle", "colab"}:
        return 2
    if sys.platform == "win32":
        return 0
    return min(2, num_workers)


def _prefetch_factor(env: str, num_workers: int) -> int | None:
    """Batches prefetched per worker: 2 on Kaggle/Colab or under DDP, else 4."""
    if num_workers <= 0:
        return None
    multi_rank = int(os.environ.get("WORLD_SIZE", "1")) > 1
    return 2 if (env in {"kaggle", "colab"} or multi_rank) else 4


def compute_worker_topology(
    env: str = "local",
    device: torch.device | None = None,
    is_forex: bool = False,
    user_workers: UserWorkerCounts | None = None,
    config: dict[str, Any] | None = None,
) -> WorkerTopology:
    """Compute optimal DataLoader worker counts, persistent worker policies, and CPU threads."""
    user = user_workers or UserWorkerCounts()
    cpu_count = os.cpu_count() or 2
    dev_type = (device or torch.device("cuda" if torch.cuda.is_available() else "cpu")).type

    ram_gb = _system_ram_gb()
    num_threads = _set_cpu_threads(env, cpu_count)

    # Determine training workers
    if user.train is not None:
        num_workers = user.train
    elif is_forex:
        # Time-series datasets stream efficiently from memory/mmap; multi-process adds IPC overhead
        num_workers = 0
    else:
        num_workers = _default_train_workers(env, ram_gb, cpu_count, config or {})

    # Determine validation workers
    if user.val is not None:
        val_num_workers = user.val
    else:
        val_num_workers = _default_val_workers(env, num_workers, config or {})

    pin_memory = dev_type == "cuda"
    persistent_workers = num_workers > 0 and ram_gb >= 16.0 and sys.platform != "win32"
    prefetch_factor = _prefetch_factor(env, num_workers)

    logger.info(
        "Worker topology computed: train_workers=%d, val_workers=%d, persistent=%s, "
        "pin_mem=%s, prefetch=%s",
        num_workers,
        val_num_workers,
        persistent_workers,
        pin_memory,
        prefetch_factor,
    )

    return WorkerTopology(
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        num_threads=num_threads,
    )


# torch keeps the live worker iterator of a DataLoader in this private attribute.
_LOADER_ITERATOR_ATTR = "_iterator"


def _shutdown_loader_iterator(loader: Any) -> None:
    """Stop the worker processes of a DataLoader's live iterator and drop the iterator.

    Private-API adapter: torch exposes no public call to stop persistent workers. The
    multiprocessing iterator only stops them in ``_shutdown_workers`` (also run by its
    ``__del__``), and the loader keeps that iterator in ``_iterator``. Clearing it lets
    the next ``iter(loader)`` start fresh workers. The names are read with ``getattr`` so
    a torch release that renames them makes this a logged no-op instead of a crash.
    """
    iterator = getattr(loader, _LOADER_ITERATOR_ATTR, None)
    if iterator is None:
        return
    shutdown_fn = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_fn):
        try:
            shutdown_fn()
        except (RuntimeError, OSError) as exc:
            logger.warning("Error shutting down DataLoader workers: %s", exc)
    else:
        logger.debug("DataLoader iterator has no _shutdown_workers; leaving workers to GC")
    setattr(loader, _LOADER_ITERATOR_ATTR, None)


def dispose_loader(loader: Any) -> None:
    """Terminate DataLoader worker processes and release pinned shared memory."""
    if loader is None:
        return

    _shutdown_loader_iterator(loader)

    # Clean dataset references if close() or shutdown() protocol exists
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except (OSError, RuntimeError) as exc:
                logger.warning("Error closing dataset resources: %s", exc)

    # Reclaim garbage and empty CUDA memory cache if available
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            logger.debug("Error releasing CUDA cache during loader disposal: %s", exc)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Worker initialization to prevent DataLoader child processes from intercepting SIGINT.

    By ignoring SIGINT in workers, the main process handles the user interruption
    signal cleanly, preventing 'RuntimeError: DataLoader worker exited unexpectedly
    with exit code 130' and BrokenPipeError during shutdown.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError) as exc:
        logger.debug("DataLoader worker %d could not ignore SIGINT: %s", worker_id, exc)
