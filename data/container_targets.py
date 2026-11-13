"""Per-task targets of container-backed (WebDataset, Parquet, MDS, LitData) samples.

Container samples carry their image, optional target and mask bytes, a label (label file
text or a JSON-derived value) and the JSON sidecar in ``metadata["json"]``. Both the
existing shard layout (labels inside the JSON sidecar) and the compiler's current layout
(``.txt`` label sidecars, lossless ``.mask.webp`` masks) are decoded here. Each builder
returns the same ``(input, target, task_id)`` triple as the directory-mode builder of its
task in :mod:`data.dataset_targets`.
"""

from __future__ import annotations

import io
import logging
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from data.dataset_targets import MAX_MASK_CLASS, RESTORATION_TASKS, multitask_head, reseed
from data.sample_labels import (
    authenticity_distribution,
    class_index,
    empty_face_target,
    face_target_from_json,
    face_target_from_text,
    ffanet_boxes,
    quality_scores,
)
from training.data.containers.base import Sample
from training.data.degrade import (
    apply_film_degradation,
    apply_synthetic_degradation,
    synthesize_degradation,
)
from training.ignored_targets import IGNORE_TARGET

logger = logging.getLogger("lemtrain.dataset")

SUPER_RESOLUTION_MODELS = {"ultrazoom": 2}
"""Model key -> upscale factor; their input is the target downscaled by that factor."""


@dataclass(frozen=True)
class DecodedSample:
    """One container sample with its images decoded and its input transformed."""

    ds_name: str
    sample: Sample
    img: Image.Image
    target_img: Image.Image | None
    img_tensor: torch.Tensor
    seed: int

    @property
    def meta_json(self) -> Any:
        """The JSON sidecar (``None`` when the sample has none)."""
        return self.sample.metadata.get("json")

    @property
    def label_text(self) -> str | None:
        """The label when it is label-file text, else ``None``."""
        return self.sample.label if isinstance(self.sample.label, str) else None


def class_index_from_file(classes_file: str, label: str) -> int | None:
    """``label``'s line index in ``classes_file``; ``None`` when absent or unreadable."""
    if not os.path.exists(classes_file):
        return None
    try:
        with open(classes_file, "r", encoding="utf-8") as cf:
            c_lines = [line.strip() for line in cf if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s for label %r: %s", classes_file, label, exc)
        return None
    return c_lines.index(label) if label in c_lines else None


class ContainerTargetsMixin:
    """Task builders for container samples, mixed into ``MultiTaskDataset``."""

    # Host (``MultiTaskDataset``) state this mixin reads.
    opts: DatasetOptions
    spec: ModelSpec
    catalog: SampleCatalog
    pipes: TransformSet

    def container_task_item(self, decoded: DecodedSample):
        """Dispatch ``decoded`` to its task type's builder."""
        if self.spec.task_type in RESTORATION_TASKS:
            return self._container_restoration_item(decoded)
        builders: dict[str, Callable[[DecodedSample], tuple]] = {
            "parameter_prediction": self._container_parameter_item,
            "quality": self._container_quality_item,
            "classification": self._container_classification_item,
            "segmentation": self._container_segmentation_item,
            "face_detection": self._container_face_item,
        }
        builder = builders.get(self.spec.task_type)
        if builder is None:
            raise RuntimeError(
                f"No container target builder for task '{self.spec.task_type}' "
                f"(model '{self.spec.model_key}', dataset '{decoded.ds_name}')."
            )
        return builder(decoded)

    def seeded_transform(self, img, seed: int):
        """Apply ``pipes.transform`` after reseeding so paired crops and flips align."""
        reseed(seed)
        return self.pipes.transform(img)

    def _warn_missing(self, ds_name: str, what: str) -> None:
        """Warn once per dataset and kind that samples lack a usable label."""
        warned: set[tuple[str, str]] = self.__dict__.setdefault("_container_warned", set())
        if (ds_name, what) in warned:
            return
        warned.add((ds_name, what))
        logger.warning(
            "Container dataset '%s' (model '%s'): samples without %s; their target is empty.",
            ds_name,
            self.spec.model_key,
            what,
        )

    # --- restoration, enhancement, face ---------------------------------------------------

    def _container_restoration_item(self, decoded: DecodedSample):
        """Restoration triple: the decoded target, or a degradation synthesised from the input.

        Unpaired samples never train on identity mappings: the clean image becomes the
        target and a synthesised degradation (film damage for ``film_restorer``) the input.
        Super-resolution models get their input downscaled from the aligned target crop.
        """
        seed = decoded.seed
        if decoded.target_img is not None:
            img_tensor = decoded.img_tensor
            target_tensor = self.seeded_transform(decoded.target_img, seed)
        elif self.spec.model_key == "film_restorer":
            target_tensor = self.seeded_transform(decoded.img, seed)
            img_tensor = apply_film_degradation(target_tensor)
        else:
            target_tensor = self.seeded_transform(decoded.img, seed)
            img_tensor = self.seeded_transform(synthesize_degradation(decoded.img), seed)

        scale = SUPER_RESOLUTION_MODELS.get(self.spec.model_key)
        if scale:
            img_tensor = _downscale(img_tensor, scale)

        task_id = self.spec.task_type
        if self.spec.model_key == "professional_multitask_restoration" and decoded.sample.name:
            task_id = multitask_head(decoded.sample.name)
        if self.spec.model_key.startswith("ffanet"):
            target = {"image": target_tensor, "bboxes": ffanet_boxes(decoded.label_text)}
            return img_tensor, target, task_id
        return img_tensor, target_tensor, task_id

    # --- parameter prediction (UPN v2) ----------------------------------------------------

    def _container_parameter_item(self, decoded: DecodedSample):
        """Degrade the clean image with random parameters; the target is the parameters."""
        clean_img = decoded.target_img if decoded.target_img is not None else decoded.img
        reseed(decoded.seed)
        clean_tensor = self.pipes.clean_transform(clean_img)
        deg = random.random()
        theta = random.random() * math.pi
        conf = random.random()
        degraded = apply_synthetic_degradation(clean_tensor, deg, theta, conf)
        params = torch.tensor([deg, theta, conf], dtype=torch.float32)
        return degraded, params, "parameter_prediction"

    # --- quality (NIMA) -------------------------------------------------------------------

    def _container_quality_item(self, decoded: DecodedSample):
        """10-bin NIMA distribution from the label, the JSON sidecar or the NIMA index."""
        scores = quality_scores(decoded.sample.label)
        if scores is None:
            scores = quality_scores(decoded.meta_json)
        if scores is None:
            mean_score = self.catalog.nima_scores.get(decoded.sample.name)
            if mean_score is not None and self.spec.model_key == "nima_authenticity":
                scores = authenticity_distribution(mean_score)
        if scores is None:
            self._warn_missing(decoded.ds_name, "a NIMA score distribution")
            scores = torch.zeros(10, dtype=torch.float32)
        return decoded.img_tensor, scores, "quality"

    # --- classification -------------------------------------------------------------------

    def _container_classification_item(self, decoded: DecodedSample):
        """Class index from a numeric label, a digit string or a ``classes.txt`` entry."""
        class_idx = self._container_class_index(decoded)
        if class_idx is None:
            # No class: ignored by the loss and the accuracy instead of training as class 0.
            self.catalog.unlabeled.record(decoded.ds_name, self.spec.model_key, decoded.sample.name)
            class_idx = IGNORE_TARGET
        return decoded.img_tensor, torch.tensor(class_idx, dtype=torch.long), "classification"

    def _container_class_index(self, decoded: DecodedSample) -> int | None:
        """Class id of the sample's label (:func:`data.sample_labels.class_index`).

        A name resolves through the registry ``class_names``; only a model without them
        falls back to the dataset's ``classes.txt``. ``ClassLabelRangeError`` for an id
        outside the registry classes.
        """
        label = decoded.sample.label
        meta = decoded.meta_json
        if label is None and isinstance(meta, dict):
            label = meta.get("label", meta.get("class", meta.get("class_id")))
        class_idx = class_index(label, self.spec.class_names)
        if class_idx is not None or self.spec.class_names or not isinstance(label, str):
            return class_idx
        classes_file = os.path.join(
            self.catalog.path_cache.get(decoded.ds_name) or "", "classes.txt"
        )
        return class_index_from_file(classes_file, label.strip())

    # --- segmentation (ParseNet) ----------------------------------------------------------

    def _container_segmentation_item(self, decoded: DecodedSample):
        """Class mask target decoded from the sample's ``.mask.*`` member.

        Raises ``RuntimeError`` when the sample has no mask: training a segmentation
        model on all-zero masks would silently learn nothing.
        """
        mask_bytes = decoded.sample.mask_bytes
        if not mask_bytes:
            raise RuntimeError(
                f"Segmentation sample '{decoded.sample.name}' in '{decoded.ds_name}' has no "
                "mask member (expected '<key>.mask.webp'); recompile the manifold with masks."
            )
        with Image.open(io.BytesIO(mask_bytes)) as raw_mask:
            mask_img = raw_mask.convert("L")
        reseed(decoded.seed)
        mask = self.pipes.mask_transform(mask_img)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return decoded.img_tensor, torch.clamp(mask_tensor, 0, MAX_MASK_CLASS), "segmentation"

    # --- face detection (RetinaFace) ------------------------------------------------------

    def _container_face_item(self, decoded: DecodedSample):
        """Confidence, box and landmarks from a YOLO label or the JSON ``objects`` list."""
        target = face_target_from_text(decoded.label_text)
        if target is None:
            target = face_target_from_json(decoded.meta_json)
        if target is None:
            target = empty_face_target()
        return decoded.img_tensor, target, "face_detection"


def _downscale(img_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """Antialiased bicubic downscale of a ``[C, H, W]`` tensor by ``scale``."""
    height, width = img_tensor.shape[-2:]
    size = (max(1, height // scale), max(1, width // scale))
    small = F.interpolate(
        img_tensor.unsqueeze(0), size=size, mode="bicubic", align_corners=False, antialias=True
    )
    return small.squeeze(0).clamp(0.0, 1.0)
