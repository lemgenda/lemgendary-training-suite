"""Per-task targets of container-backed (WebDataset, Parquet, MDS, LitData) samples.

Container samples carry their image, optional target and mask bytes (lossless
``.mask.webp`` masks), a label value and their JSON label document in
``metadata["json"]``. A shard sample's labels come only from that document
(:mod:`data.json_labels`): boxes from ``objects``, classes from ``category_id`` or a
``class_name`` mapped through the manifold's ``dataset_info.yaml`` ``classes``. An
unreadable or malformed document raises :class:`~data.json_labels.LabelDocumentError`
instead of training as an empty target. Each builder returns the same
``(input, target, task_id)`` triple as the directory-mode builder of its task in
:mod:`data.dataset_targets`.
"""

from __future__ import annotations

import io
import logging
import math
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from data.dataset_targets import MAX_MASK_CLASS, RESTORATION_TASKS, multitask_head, reseed
from data.json_labels import (
    LabelDocumentError,
    class_label,
    object_boxes,
    sample_document,
    yolo_rows,
)
from data.manifold_classes import manifold_classes
from data.sample_labels import (
    authenticity_distribution,
    box_tensor,
    class_index,
    empty_face_target,
    face_target_from_json,
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

    def label_document(self) -> Mapping[str, Any] | None:
        """The sample's JSON label document; ``None`` when it has no ``.json`` member.

        Raises :class:`~data.json_labels.LabelDocumentError` for an unreadable document.
        """
        return sample_document(
            self.sample.metadata, f"sample '{self.sample.name}' in '{self.ds_name}'"
        )


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
            target = {"image": target_tensor, "bboxes": self._ffanet_boxes(decoded)}
            return img_tensor, target, task_id
        return img_tensor, target_tensor, task_id

    def _ffanet_boxes(self, decoded: DecodedSample) -> torch.Tensor:
        """FFA-Net boxes from the JSON ``objects``; none (warned once) without ``objects``."""
        doc = decoded.label_document()
        rows = None
        if doc is not None:
            rows = self._json_rows(decoded, doc)
        if rows is None:
            self._warn_missing(decoded.ds_name, "a JSON objects label")
            rows = []
        return box_tensor(rows)

    def _json_rows(self, decoded: DecodedSample, doc: Mapping[str, Any]):
        """YOLO rows of ``doc``; a schema error names the sample."""
        try:
            return yolo_rows(doc, self.dataset_classes(decoded.ds_name))
        except LabelDocumentError as exc:
            raise LabelDocumentError(
                f"sample '{decoded.sample.name}' in '{decoded.ds_name}': {exc}"
            ) from exc

    def dataset_classes(self, ds_name: str) -> tuple[str, ...]:
        """The manifold's ``dataset_info.yaml`` ``classes`` (read once per dataset)."""
        cache = self.catalog.dataset_classes
        if ds_name not in cache:
            cache[ds_name] = manifold_classes(self.catalog.path_cache.get(ds_name) or "")
        return cache[ds_name]

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
        """10-bin NIMA distribution from the JSON label, the sample label or the NIMA index."""
        scores = quality_scores(decoded.label_document())
        if scores is None:
            scores = quality_scores(decoded.sample.label)
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

        The label is the JSON ``class_id`` / ``class_name`` (then the reader's label value).
        A name resolves through the registry ``class_names``; only a model without them
        falls back to the manifold's ``dataset_info.yaml`` ``classes``, then its
        ``classes.txt``. ``ClassLabelRangeError`` for an id outside the registry classes.
        """
        label = class_label(decoded.label_document())
        if label is None:
            label = decoded.sample.label
        class_idx = class_index(label, self.spec.class_names)
        if class_idx is not None or self.spec.class_names or not isinstance(label, str):
            return class_idx
        name = label.strip()
        classes = self.dataset_classes(decoded.ds_name)
        if name in classes:
            return classes.index(name)
        classes_file = os.path.join(
            self.catalog.path_cache.get(decoded.ds_name) or "", "classes.txt"
        )
        return class_index_from_file(classes_file, name)

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
        """Confidence, box and landmarks from the JSON ``objects`` list.

        An empty ``objects`` list is an image without a face. Raises
        :class:`~data.json_labels.LabelDocumentError` when the sample has no JSON label or
        the label has no ``objects`` (every face sample carries one; a missing one is a
        compile defect, not a negative).
        """
        doc = decoded.label_document()
        where = f"Face sample '{decoded.sample.name}' in '{decoded.ds_name}'"
        if doc is None:
            raise LabelDocumentError(
                f"{where} has no JSON label (expected '<key>.json' with objects); "
                "recompile or repair the manifold."
            )
        try:
            boxes = object_boxes(doc)
        except LabelDocumentError as exc:
            raise LabelDocumentError(f"{where}: {exc}") from exc
        if boxes is None:
            raise LabelDocumentError(f"{where}: the JSON label has no objects list")
        target = face_target_from_json(doc)
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
