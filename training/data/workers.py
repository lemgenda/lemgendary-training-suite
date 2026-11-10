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


def _available_ram_gb() -> float:
    """Available system RAM in GB (8.0 when psutil cannot tell)."""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except OSError as exc:
        logger.debug("psutil unavailable for available memory calculation: %s", exc)
        return 8.0


def _set_cpu_threads(env: str, cpu_count: int) -> int:
    """Set torch CPU threads (1 on Kaggle, else every core); return the count."""
    num_threads = 1 if env == "kaggle" else max(1, cpu_count)
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError as exc:
        logger.debug("Failed setting torch CPU threads to %d: %s", num_threads, exc)
    return num_threads


def _default_train_workers(
    env: str,
    ram_gb: float,
    available_ram_gb: float,
    cpu_count: int,
    cfg: dict[str, Any],
) -> int:
    """Dynamically determine training DataLoader workers based on live resources.

    Allocates workers proportional to CPU cores and available RAM headroom to maximize
    disk/decompression throughput without risking host pagefile thrashing.
    """
    if env in {"kaggle", "colab"}:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # In multi-GPU DDP on Kaggle (2 GPUs), 4 workers per process spawns 8 total workers,
        # exhausting shared memory and RAM. Cap to 2 workers per process in DDP.
        return 2 if world_size > 1 else min(cpu_count, 4)

    cfg_workers = cfg.get("hardware", {}).get("num_workers", "auto")
    if isinstance(cfg_workers, int) and cfg_workers >= 0:
        return min(cpu_count, cfg_workers)

    # Dynamic resource sizing:
    # 1. CPU Core Headroom: leave 1 core for OS / main training process
    core_limit = max(1, cpu_count - 1)

    # 2. Memory Headroom: budget ~1.5 GB available RAM per worker process
    mem_limit = max(0, int(available_ram_gb // 1.5))

    # 3. Safe upper bound per environment
    if sys.platform == "win32":
        # On Windows, spawning processes incurs more memory overhead than Linux fork.
        # Dynamically scale up to 6 workers if RAM and cores allow; scale down if RAM is low.
        if ram_gb < 8.0 or available_ram_gb < 2.0:
            return 0
        if ram_gb < 16.0 or available_ram_gb < 4.0:
            return min(core_limit, mem_limit, 2)
        return max(2, min(core_limit, mem_limit, 6))

    # Linux / macOS
    return max(1, min(core_limit, mem_limit, 8))


def _default_val_workers(env: str, num_workers: int, cfg: dict[str, Any]) -> int:
    """Validation workers: ``hardware.val_num_workers`` if an int, else dynamically scaled."""
    cfg_val = cfg.get("hardware", {}).get("val_num_workers", "auto")
    if isinstance(cfg_val, int) and cfg_val >= 0:
        return cfg_val
    if env in {"kaggle", "colab"}:
        return 2
    if num_workers == 0:
        return 0
    # Scale validation workers proportionally to avoid stalling evaluation
    return min(2, max(1, num_workers // 2))


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
    available_ram_gb = _available_ram_gb()
    num_threads = _set_cpu_threads(env, cpu_count)

    # Determine training workers
    if user.train is not None:
        num_workers = user.train
    elif is_forex:
        # Time-series datasets stream efficiently from memory/mmap; multi-process adds IPC overhead
        num_workers = 0
    else:
        num_workers = _default_train_workers(
            env=env,
            ram_gb=ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_count=cpu_count,
            cfg=config or {},
        )

    # Determine validation workers
    if user.val is not None:
        val_num_workers = user.val
    else:
        val_num_workers = _default_val_workers(env, num_workers, config or {})

    pin_memory = dev_type == "cuda"
    persistent_workers = num_workers > 0 and ram_gb >= 16.0 and available_ram_gb >= 3.0
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
    release_cuda_memory("loader disposal")


def release_cuda_memory(purpose: str) -> None:
    """Collect garbage and empty the CUDA cache; a failed release is logged, not raised."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            logger.debug("CUDA cache release failed during %s: %s", purpose, exc)


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
