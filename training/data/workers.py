"""Worker topology calculation and memory-leak-free DataLoader disposal."""

from dataclasses import dataclass
import gc
import logging
import os
import sys
from typing import Any
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


def compute_worker_topology(
    env: str = "local",
    device: torch.device | None = None,
    is_forex: bool = False,
    user_num_workers: int | None = None,
    user_val_workers: int | None = None,
    config: dict[str, Any] | None = None,
) -> WorkerTopology:
    """Compute optimal DataLoader worker counts, persistent worker policies, and CPU threads."""
    cfg = config or {}
    cpu_count = os.cpu_count() or 2
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probe system memory
    ram_gb = 16.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except (ImportError, OSError) as exc:
        logger.debug("psutil unavailable for worker topology calculation: %s", exc)

    # Determine CPU thread allotment
    if env == "kaggle":
        num_threads = 1
    else:
        num_threads = max(1, cpu_count)

    try:
        torch.set_num_threads(num_threads)
    except RuntimeError as exc:
        logger.debug("Failed setting torch CPU threads to %d: %s", num_threads, exc)

    # Determine training workers
    if user_num_workers is not None:
        num_workers = user_num_workers
    elif is_forex:
        # Time-series datasets stream efficiently from memory/mmap; multi-process adds IPC overhead
        num_workers = 0
    elif env in {"kaggle", "colab"}:
        num_workers = 4
    elif sys.platform == "win32":
        # Windows multi-process PyTorch spawns via freeze_support; restrict to 2 if ample RAM
        num_workers = 2 if ram_gb >= 16.0 else 0
    else:
        cfg_workers = cfg.get("hardware", {}).get("num_workers", 4)
        if isinstance(cfg_workers, int):
            num_workers = min(cpu_count, cfg_workers)
        else:
            num_workers = min(cpu_count, 4)

    # Determine validation workers
    if user_val_workers is not None:
        val_num_workers = user_val_workers
    else:
        cfg_val = cfg.get("hardware", {}).get("val_num_workers", "auto")
        if isinstance(cfg_val, int):
            val_num_workers = cfg_val
        elif env in {"kaggle", "colab"}:
            val_num_workers = 2
        elif sys.platform == "win32":
            val_num_workers = 0
        else:
            val_num_workers = min(2, num_workers)

    pin_memory = dev.type == "cuda"
    persistent_workers = num_workers > 0 and ram_gb >= 16.0 and sys.platform != "win32"
    prefetch_factor = 8 if num_workers > 0 else None

    logger.info(
        "Worker topology computed: train_workers=%d, val_workers=%d, persistent=%s, pin_mem=%s, prefetch=%s",
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


def dispose_loader(loader: Any) -> None:
    """Cleanly terminate DataLoader worker processes and release pinned shared memory without suppression."""
    if loader is None:
        return

    # Check for active _iterator attached to the DataLoader
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown_fn = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except (RuntimeError, OSError) as exc:
                logger.warning("Error shutting down DataLoader workers: %s", exc)
        setattr(loader, "_iterator", None)

    # Clean dataset references if close() or shutdown() protocol exists
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("Error closing dataset resources: %s", exc)

    # Reclaim garbage and empty CUDA memory cache if available
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            logger.debug("Error releasing CUDA cache during loader disposal: %s", exc)
