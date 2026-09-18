"""Data ingestion, loaders, and container readers package."""

from training.data.degrade import (
    DynamicOnTheFlyDegrader,
    JpegCompressionGuard,
    apply_film_degradation,
    apply_synthetic_degradation,
    synthesize_degradation,
)
from training.data.loaders import build_train_loader, build_val_loader, rebuild_train_loader
from training.data.manifold import ManifoldInfo, ManifoldResolver
from training.data.workers import WorkerTopology, compute_worker_topology, dispose_loader

__all__ = [
    "DynamicOnTheFlyDegrader",
    "JpegCompressionGuard",
    "ManifoldInfo",
    "ManifoldResolver",
    "WorkerTopology",
    "apply_film_degradation",
    "apply_synthetic_degradation",
    "build_train_loader",
    "build_val_loader",
    "compute_worker_topology",
    "dispose_loader",
    "rebuild_train_loader",
    "synthesize_degradation",
]
