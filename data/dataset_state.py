"""Grouped host state for ``MultiTaskDataset`` and its mixins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetOptions:
    """Split, environment and curriculum options of a dataset instance."""

    is_train: bool
    env: str
    split: str
    data_root: str
    sample_fraction: float
    size: tuple[int, int]
    stress: float = 0.0  # 2026 Resilience: Dynamic Stress Protocol


@dataclass
class ModelSpec:
    """The registry entry (``unified_models``) the dataset serves."""

    unified_models: dict
    model_key: str
    model_info: dict
    task_type: str


@dataclass
class SampleCatalog:
    """Resolved dataset paths, the sample index and per-dataset readers and scores."""

    samples: list[tuple[str, str | int]] = field(default_factory=list)
    all_samples: list[tuple[str, str | int]] = field(default_factory=list)
    path_cache: dict[str, str | None] = field(default_factory=dict)
    container_readers: dict[str, Any] = field(default_factory=dict)
    nima_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TransformSet:
    """Image, mask and clean (parameter-prediction) transforms."""

    transform: Any = None
    mask_transform: Any = None
    clean_transform: Any = None
