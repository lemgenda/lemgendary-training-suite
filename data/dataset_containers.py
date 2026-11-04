"""Container-backed samples for MultiTaskDataset.

Packed-container (WebDataset, Parquet, MDS, LitData) loading, item decoding and DDP sync
dummies.
"""

from __future__ import annotations

import io
import logging
import os
import random

import torch
from PIL import Image

from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from training.data.containers import resolve_container_reader
from training.data.degrade import (
    synthesize_degradation,
)

logger = logging.getLogger("lemtrain.dataset")


def _class_index_from_file(classes_file: str, label: str) -> int:
    """Return ``label``'s line index in ``classes_file`` (0 when absent or unreadable)."""
    class_idx = 0
    if os.path.exists(classes_file):
        try:
            with open(classes_file, "r", encoding="utf-8") as cf:
                c_lines = [line.strip() for line in cf if line.strip()]
            if label in c_lines:
                class_idx = c_lines.index(label)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "Could not read %s; using class index 0 for %r: %s",
                classes_file,
                label,
                exc,
            )
    return class_idx


def _quality_scores(label) -> torch.Tensor:
    """The first ten NIMA bins of a list label; zeros for scalar or missing labels."""
    if isinstance(label, (list, tuple)) and len(label) >= 10:
        return torch.tensor(label[:10], dtype=torch.float32)
    return torch.zeros(10, dtype=torch.float32)


class DatasetContainerMixin:
    """Container-backed samples mixed into ``MultiTaskDataset``."""

    # Host (``MultiTaskDataset``) state this mixin reads and writes.
    opts: DatasetOptions
    spec: ModelSpec
    catalog: SampleCatalog
    pipes: TransformSet

    def load_container_samples(self, loaded_paths: set) -> None:
        """Attempt container-format loading when no directory images are found.

        Iterates over already-resolved dataset paths and uses ``resolve_container_reader``
        to detect WebDataset, Parquet, MDS, or LitData layouts. Builds a synthetic
        ``all_samples`` index as ``(ds_name, int_index)`` tuples and stores an active
        ``ContainerReader`` per dataset in ``self.catalog.container_readers``.
        """
        for ds_name, ds_path in self.catalog.path_cache.items():
            if ds_path is None or ds_path not in loaded_paths:
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

    def getitem_container(self, ds_name: str, idx: int):
        """Fetch and decode a sample from a ContainerReader-backed dataset."""
        reader = self.catalog.container_readers[ds_name]
        try:
            sample = reader[idx]
        except (IndexError, OSError, RuntimeError) as exc:
            logger.debug("Container read failed for %s[%d]: %s", ds_name, idx, exc)
            return self.sync_dummy()

        # Decode image bytes.
        try:
            img = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        except (OSError, ValueError) as dec_err:
            logger.debug("Image decode failed for %s[%d]: %s", ds_name, idx, dec_err)
            img = Image.new("RGB", self.opts.size, (128, 128, 128))

        # Decode target bytes if present.
        target_img = None
        if sample.target_bytes:
            try:
                target_img = Image.open(io.BytesIO(sample.target_bytes)).convert("RGB")
            except (OSError, ValueError) as tdec_err:
                logger.debug("Target decode failed for %s[%d]: %s", ds_name, idx, tdec_err)

        seed = random.randint(0, 2**32 - 1)
        img_tensor = self._seeded_transform(img, seed)
        task_type = self.spec.task_type

        if task_type in ("restoration", "enhancement", "face"):
            return self._container_restoration_item(img, target_img, img_tensor, seed)
        if task_type == "quality":
            return img_tensor, _quality_scores(sample.label), "quality"
        if task_type == "classification":
            class_idx = self._container_class_index(ds_name, sample.label)
            return img_tensor, torch.tensor(class_idx, dtype=torch.long), "classification"
        return img_tensor, torch.zeros(1), task_type

    def _seeded_transform(self, img, seed: int):
        """Apply ``pipes.transform`` after reseeding so paired crops and flips align."""
        random.seed(seed)
        torch.manual_seed(seed)
        return self.pipes.transform(img)

    def _container_restoration_item(self, img, target_img, img_tensor, seed: int):
        """Restoration triple: decoded target, a synthesized degradation or the input itself."""
        model_info = self.spec.unified_models.get(self.spec.model_key, {})
        needs_syn = getattr(self, "needs_synthesis", False) or model_info.get(
            "needs_synthesis", False
        )

        if target_img is not None:
            target_tensor = self._seeded_transform(target_img, seed)
        elif needs_syn or self.spec.task_type in ("restoration", "face"):
            # Clean container sample: synthesize realistic input degradation on-the-fly
            clean_target = img
            degraded_input = synthesize_degradation(clean_target)
            target_tensor = self._seeded_transform(clean_target, seed)
            img_tensor = self._seeded_transform(degraded_input, seed)
        else:
            target_tensor = self._seeded_transform(img, seed)
        return img_tensor, target_tensor, self.spec.task_type

    def _container_class_index(self, ds_name: str, label) -> int:
        """Class index from a numeric label, a digit string or a ``classes.txt`` entry."""
        if isinstance(label, (int, float)):
            return int(label)
        if not isinstance(label, str):
            return 0
        if label.isdigit():
            return int(label)
        classes_file = os.path.join(self.catalog.path_cache.get(ds_name) or "", "classes.txt")
        return _class_index_from_file(classes_file, label)

    def sync_dummy(self):
        """Zero-filled sample with the model's input and target shapes, used in sync mode."""
        # 1. Determine input shape
        if self.spec.model_key == "ultrazoom":
            in_shape = (3, self.opts.size[0] // 2, self.opts.size[1] // 2)
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
        else:
            target_shape = (1,)
            task_str = self.spec.task_type

        return torch.zeros(in_shape), torch.zeros(target_shape), task_str
