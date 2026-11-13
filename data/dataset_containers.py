"""Container-backed samples for MultiTaskDataset.

Packed-container (WebDataset, Parquet, MDS, LitData) loading, item decoding and DDP sync
dummies.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL import Image

from data.container_targets import SUPER_RESOLUTION_MODELS, DecodedSample
from data.dataset_sources import claim_manifold, real_manifold_path
from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from data.sample_labels import FACE_TARGET_LEN
from data.sample_origin import describe_sample_origin
from training.data.containers import resolve_container_reader

logger = logging.getLogger("lemtrain.dataset")

ShardKey = tuple[str, str | None]
"""``(ds_name, shard path)`` of a sample; ``None`` when it has no shard."""


class DatasetContainerMixin:
    """Container-backed samples mixed into ``MultiTaskDataset``."""

    # Host (``MultiTaskDataset``) state this mixin reads and writes.
    opts: DatasetOptions
    spec: ModelSpec
    catalog: SampleCatalog
    pipes: TransformSet
    # Provided by ``ContainerTargetsMixin``.
    container_task_item: Callable[[DecodedSample], tuple]
    seeded_transform: Callable[[Any, int], Any]

    def load_container_samples(self, loaded_paths: set) -> None:
        """Attempt container-format loading when no directory images are found.

        ``loaded_paths`` holds the real paths of the discovered manifolds. Iterates over
        already-resolved dataset paths (one reader per real folder, so alias names never
        count a manifold twice) and uses ``resolve_container_reader``
        to detect WebDataset, Parquet, MDS, or LitData layouts. Builds a synthetic
        ``all_samples`` index as ``(ds_name, int_index)`` tuples and stores an active
        ``ContainerReader`` per dataset in ``self.catalog.container_readers``.
        """
        opened: set[str] = set()
        for ds_name, ds_path in self.catalog.path_cache.items():
            if ds_path is None or real_manifold_path(ds_path) not in loaded_paths:
                continue
            if not claim_manifold(opened, ds_name, ds_path):
                continue
            try:
                reader = resolve_container_reader(ds_path, split=self.opts.split)
                count = len(reader)
                if count == 0:
                    reader.close()
                    continue
                self.catalog.container_readers[ds_name] = reader
                for idx in range(count):
                    self.catalog.all_samples.append((ds_name, idx))
                logger.info(
                    "Container reader resolved for '%s': %s (%d samples)",
                    ds_name,
                    type(reader).__name__,
                    count,
                )
                if os.environ.get("RANK", "0") == "0":
                    print(
                        f"[CONTAINER] Resolved {type(reader).__name__} for '{ds_name}': "
                        f"{count} samples ({self.opts.split} split)"
                    )
            except (FileNotFoundError, OSError, ValueError) as exc:
                logger.debug("Container reader failed for '%s': %s", ds_name, exc)
                if self.opts.split == "train":
                    print(
                        f"[CONTAINER] Notice: Container reader for '{ds_name}' "
                        f"({self.opts.split} split): {exc}"
                    )

    def shard_keys_for(self, samples: Sequence[tuple]) -> list[ShardKey]:
        """Shard of each ``(ds_name, sample_id)``, for locality-aware orderings.

        Container samples whose reader reports shards (``shard_of``) map to
        ``(ds_name, shard)``; every other sample of a dataset (loose files, directory
        images, readers without shards) shares the key ``(ds_name, None)``.
        """
        keys: list[ShardKey] = []
        for ds_name, sample_id in samples:
            shard_of = getattr(self.catalog.container_readers.get(ds_name), "shard_of", None)
            shard = (
                shard_of(sample_id) if callable(shard_of) and isinstance(sample_id, int) else None
            )
            keys.append((ds_name, None if shard is None else str(shard)))
        return keys

    def sample_shard_keys(self) -> list[ShardKey]:
        """Shard of each active sample, in loader index order."""
        return self.shard_keys_for(self.catalog.samples)

    def getitem_container(self, ds_name: str, idx: int, seed: int):
        """Fetch, decode and build the task triple of a ContainerReader-backed sample.

        ``seed`` is the sample's augmentation seed; it aligns the paired crops and flips.
        """
        reader = self.catalog.container_readers[ds_name]
        try:
            sample = reader[idx]
        except (IndexError, OSError, RuntimeError) as exc:
            logger.warning("Container read failed for %s[%d]: %s", ds_name, idx, exc)
            return self.sync_dummy()

        try:
            img = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        except (OSError, ValueError) as dec_err:
            logger.warning(
                "Image decode failed for %s[%d] (%s); substituting neutral gray: %s",
                ds_name,
                idx,
                describe_sample_origin(reader, idx, sample, sample.image_bytes),
                dec_err,
            )
            img = Image.new("RGB", (self.opts.size[1], self.opts.size[0]), (128, 128, 128))

        target_img = None
        if sample.target_bytes:
            try:
                target_img = Image.open(io.BytesIO(sample.target_bytes)).convert("RGB")
            except (OSError, ValueError) as tdec_err:
                logger.warning(
                    "Target decode failed for %s[%d] (%s): %s",
                    ds_name,
                    idx,
                    describe_sample_origin(reader, idx, sample, sample.target_bytes),
                    tdec_err,
                )

        decoded = DecodedSample(
            ds_name=ds_name,
            sample=sample,
            img=img,
            target_img=target_img,
            img_tensor=self.seeded_transform(img, seed),
            seed=seed,
        )
        return self.container_task_item(decoded)

    def sync_dummy(self):
        """Zero-filled sample with the model's input and target shapes, used in sync mode."""
        # 1. Determine input shape
        scale = SUPER_RESOLUTION_MODELS.get(self.spec.model_key)
        if scale:
            in_shape = (3, self.opts.size[0] // scale, self.opts.size[1] // scale)
        else:
            in_shape = (3, self.opts.size[0], self.opts.size[1])

        # 2. Determine target shape and task string
        if self.spec.task_type in ["restoration", "enhancement", "face"]:
            target_shape = (3, self.opts.size[0], self.opts.size[1])
            task_str = "denoise"
        elif self.spec.task_type == "parameter_prediction":
            target_shape = (3,)
            task_str = "parameter_prediction"
        elif self.spec.task_type == "quality":
            target_shape = (10,)
            task_str = "quality"
        elif self.spec.task_type == "segmentation":
            target_shape = (self.opts.size[0], self.opts.size[1])
            task_str = "segmentation"
        elif self.spec.task_type == "face_detection":
            target_shape = (FACE_TARGET_LEN,)
            task_str = "face_detection"
        elif self.spec.task_type == "classification":
            return torch.zeros(in_shape), torch.tensor(0, dtype=torch.long), "classification"
        else:
            target_shape = (1,)
            task_str = self.spec.task_type

        return torch.zeros(in_shape), torch.zeros(target_shape), task_str
