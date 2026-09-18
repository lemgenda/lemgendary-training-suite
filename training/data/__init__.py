"""Data ingestion, loaders, and container readers package."""

from training.data.degrade import (
    DynamicOnTheFlyDegrader,
    JpegCompressionGuard,
    apply_film_degradation,
    apply_synthetic_degradation,
    synthesize_degradation,
)
from training.data.containers import (
    ContainerReader,
    DirectoryReader,
    LitDataReader,
    MdsReader,
    ParquetReader,
    Sample,
    WebDatasetReader,
    resolve_container_reader,
)
from training.data.loaders import build_train_loader, build_val_loader, rebuild_train_loader
from training.data.manifold import ManifoldInfo, ManifoldResolver
from training.data.workers import WorkerTopology, compute_worker_topology, dispose_loader

__all__ = [
    "ContainerReader",
    "DirectoryReader",
    "DynamicOnTheFlyDegrader",
    "JpegCompressionGuard",
    "LitDataReader",
    "ManifoldInfo",
    "ManifoldResolver",
    "MdsReader",
    "ParquetReader",
    "Sample",
    "WebDatasetReader",
    "WorkerTopology",
    "apply_film_degradation",
    "apply_synthetic_degradation",
    "build_train_loader",
    "build_val_loader",
    "compute_worker_topology",
    "dispose_loader",
    "rebuild_train_loader",
    "resolve_container_reader",
    "synthesize_degradation",
]
