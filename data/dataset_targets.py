"""Directory-backed sample loading and per-task targets for MultiTaskDataset.

``load_directory_sample`` resolves a sample's files, loads the input image and, for
restoration-type tasks, the target image. One builder per task type then turns the loaded
sample into the ``(input, target, task_id)`` triple that ``__getitem__`` returns.
"""

from __future__ import annotations

import logging
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from training.data.degrade import (
    apply_film_degradation,
    apply_synthetic_degradation,
    synthesize_degradation,
)

logger = logging.getLogger("lemtrain.dataset")

RESTORATION_TASKS = ("restoration", "enhancement", "face")

# Filename substrings mapped to the multitask restorer's task heads. Order matters: the
# first entry with a matching substring wins.
_MULTITASK_HEADS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("denois",), "denoise"),
    (("deblur",), "deblur"),
    (("derain",), "derain"),
    (("dehaze_outdoor", "outdoor"), "dehaze_outdoor"),
    (("dehaze_indoor", "indoor", "dehaze", "ffa"), "dehaze_indoor"),
    (("lowlight",), "lowlight"),
    (("exposure",), "exposure"),
    (("zoom", "super"), "superres"),
    (("vintage", "film"), "vintage"),
    (("codeformer", "face_restor"), "face_restorer"),
    (("parsenet", "face_pars"), "face_parser"),
)

_MAX_FFANET_BOXES = 50
_MAX_MASK_CLASS = 18


def multitask_head(fname: str) -> str:
    """Map a manifold filename to a multitask restorer head (``denoise`` by default)."""
    fname_lower = fname.lower()
    for needles, head in _MULTITASK_HEADS:
        if any(needle in fname_lower for needle in needles):
            return head
    return "denoise"


def reseed(seed: int) -> None:
    """Seed ``random`` and torch so paired transforms crop and flip identically."""
    random.seed(seed)
    torch.manual_seed(seed)


def read_text(path: str) -> str | None:
    """Return the file's text, or ``None`` when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_lines(path: str) -> list[str]:
    """Return the file's lines, or an empty list when it does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(f)


@dataclass(frozen=True)
class SamplePaths:
    """Where a directory-backed sample lives: its name, dataset root, input and target."""

    fname: str
    ds_path: str
    img_path: str
    tgt_path: str


@dataclass
class DirectorySample:
    """A directory-backed sample: its paths, loaded input and seeded tensors."""

    paths: SamplePaths
    task_str: str
    img: Any = None
    seed: int = 0
    img_tensor: Any = None
    target_tensor: Any = None


class DatasetTargetsMixin:
    """Directory sample loading and per-task target builders mixed into ``MultiTaskDataset``."""

    # Host (``MultiTaskDataset``) state this mixin reads.
    opts: DatasetOptions
    spec: ModelSpec
    catalog: SampleCatalog
    pipes: TransformSet
    # Provided by the host and by ``DatasetSourcesMixin``.
    load_image: Callable[[str], Image.Image]
    get_dataset_path: Callable[[str], str | None]
    # ``split_path(ds_path, folder_name, fname, ext)``
    split_path: Callable[[str, str, str, str], str]

    def load_directory_sample(self, ds_name: str, fname: str) -> DirectorySample:
        """Resolve ``fname`` in ``ds_name``, load its input and (restoration) target tensors."""
        ds_path = self.catalog.path_cache.get(ds_name)
        if ds_path is None:
            ds_path = self.get_dataset_path(ds_name)
            self.catalog.path_cache[ds_name] = ds_path
        if ds_path is None:
            ds_path = ""

        sample = DirectorySample(
            paths=SamplePaths(
                fname=fname,
                ds_path=ds_path,
                img_path=self.split_path(ds_path, "images", fname, ""),
                tgt_path=self.split_path(ds_path, "targets", fname, ""),
            ),
            task_str=self.spec.task_type,
        )
        has_img = os.path.exists(sample.paths.img_path)
        has_tgt = os.path.exists(sample.paths.tgt_path)
        if self.spec.model_key == "professional_multitask_restoration":
            sample.task_str = multitask_head(fname)

        restoration = self.spec.task_type in RESTORATION_TASKS
        if not has_img and has_tgt and restoration:
            self._load_synthesized_pair(sample)
        else:
            self._load_input(sample)
            if restoration:
                sample.target_tensor = self._restoration_target(sample, has_tgt)
            else:
                sample.target_tensor = sample.img_tensor  # Fallback initialization
        return sample

    def _load_synthesized_pair(self, sample: DirectorySample) -> None:
        """Degrade the clean target into the input; one seed aligns both crops and flips."""
        target = self.load_image(sample.paths.tgt_path)
        sample.img = synthesize_degradation(target)
        sample.seed = random.randint(0, 2**32 - 1)
        reseed(sample.seed)
        sample.target_tensor = self.pipes.transform(target)
        reseed(sample.seed)
        sample.img_tensor = self.pipes.transform(sample.img)

    def _load_input(self, sample: DirectorySample) -> None:
        """Load the input image; ``ultrazoom`` gets an on-the-fly 2x downscaled input."""
        sample.img = self.load_image(sample.paths.img_path)
        sample.seed = random.randint(0, 2**32 - 1)
        if self.spec.model_key == "ultrazoom":
            lr_size = (self.opts.size[0] // 2, self.opts.size[1] // 2)
            input_transform = transforms.Compose(
                [
                    transforms.Resize(lr_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
        else:
            input_transform = self.pipes.transform
        reseed(sample.seed)
        sample.img_tensor = input_transform(sample.img)

    def _restoration_target(self, sample: DirectorySample, has_tgt: bool):
        """Return the restoration target: a parsing mask, the target image or the input."""
        if sample.task_str == "face_parser":
            return self.load_mask(sample)
        source = self.load_image(sample.paths.tgt_path) if has_tgt else sample.img
        reseed(sample.seed)
        return self.pipes.transform(source)  # Without a target, the input keeps the HR shape.

    def load_mask(self, sample: DirectorySample):
        """Load the sample's class mask (``masks/``), clamped to valid classes, or zeros."""
        mask_path = self.split_path(sample.paths.ds_path, "masks", sample.paths.fname, "")
        if not os.path.exists(mask_path):
            return torch.zeros((self.opts.size[0], self.opts.size[1]), dtype=torch.long)
        mask = Image.open(mask_path).convert("L")  # Masks are grayscale class labels.
        reseed(sample.seed)
        mask = self.pipes.mask_transform(mask)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        # Clamp so garbage dataset masks cannot crash NLLLoss2d.
        return torch.clamp(mask_tensor, 0, _MAX_MASK_CLASS)

    def label_path(self, sample: DirectorySample) -> str:
        """Return the sample's ``labels/<name>.txt`` path."""
        return self.split_path(sample.paths.ds_path, "labels", sample.paths.fname, ".txt")

    def restoration_item(self, sample: DirectorySample):
        """Restoration, enhancement and face items (film degradation, FFA-Net boxes)."""
        img_tensor = sample.img_tensor
        if self.spec.model_key == "film_restorer":
            # An identical input and target file means a clean dataset such as DIV2K.
            try:
                if os.path.exists(sample.paths.tgt_path) and os.path.samefile(
                    sample.paths.img_path, sample.paths.tgt_path
                ):
                    img_tensor = apply_film_degradation(sample.target_tensor)
            except OSError as exc:
                logger.debug(
                    "samefile check failed for %s vs %s: %s",
                    sample.paths.img_path,
                    sample.paths.tgt_path,
                    exc,
                )

        if self.spec.model_key.startswith("ffanet"):
            target = {"image": sample.target_tensor, "bboxes": self._ffanet_boxes(sample)}
            return img_tensor, target, sample.task_str
        return img_tensor, sample.target_tensor, sample.task_str

    def _ffanet_boxes(self, sample: DirectorySample):
        """Read up to 50 five-value boxes from the label file, zero-padded."""
        bboxes = []
        for line in read_lines(self.label_path(sample)):
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes.append([float(x) for x in parts[:5]])

        bboxes_tensor = torch.zeros((_MAX_FFANET_BOXES, 5), dtype=torch.float32)
        if bboxes:
            num_boxes = min(len(bboxes), _MAX_FFANET_BOXES)
            bboxes_tensor[:num_boxes] = torch.tensor(bboxes[:num_boxes], dtype=torch.float32)
        return bboxes_tensor

    def parameter_item(self, sample: DirectorySample):
        """Degrade the clean image with random parameters; the target is the parameters."""
        tgt_path = self.split_path(sample.paths.ds_path, "targets", sample.paths.fname, "")
        clean_img = self.load_image(tgt_path) if os.path.exists(tgt_path) else sample.img
        clean_tensor = self.pipes.clean_transform(clean_img)

        deg = random.random()  # [0.0, 1.0]
        theta = random.random() * math.pi  # [0.0, pi]
        conf = random.random()  # [0.0, 1.0]
        degraded_tensor = apply_synthetic_degradation(clean_tensor, deg, theta, conf)
        params = torch.tensor([deg, theta, conf], dtype=torch.float32)
        return degraded_tensor, params, "parameter_prediction"

    def quality_item(self, sample: DirectorySample):
        """NIMA items: a 10-bin score distribution from the label file or the NIMA index."""
        if self.spec.model_key == "nima_authenticity":
            # 'real' peaks at bin 10 (class 1); 'shutterstock' peaks at bin 1 (class 0).
            n_score = self.catalog.nima_scores.get(os.path.splitext(sample.paths.fname)[0], 0.0)
            if n_score > 5.0:
                score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
            else:
                score = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return sample.img_tensor, torch.tensor(score, dtype=torch.float32), "quality"

        label_path = self.label_path(sample)
        try:
            text = read_text(label_path)
            if text is not None:
                score = [float(x) for x in text.split()]
                if len(score) < 10:
                    score = score + [0.0] * (10 - len(score))
                return sample.img_tensor, torch.tensor(score[:10], dtype=torch.float32), "quality"
        except (ValueError, TypeError) as lbl_err:
            logger.debug("Invalid quality label in %s: %s", label_path, lbl_err)
        return sample.img_tensor, torch.zeros(10), "quality"

    def classification_item(self, sample: DirectorySample):
        """Class index from the label file, a numeric parent folder or ``classes.txt``."""
        class_idx = self._label_class(sample)
        if class_idx is None:
            class_idx = self._folder_class(sample)
        return sample.img_tensor, torch.tensor(class_idx, dtype=torch.long), "classification"

    def _label_class(self, sample: DirectorySample) -> int | None:
        """Class index from ``labels/<name>.txt``, or ``None``."""
        label_path = self.label_path(sample)
        try:
            text = read_text(label_path)
            return None if text is None else int(float(text.strip()))
        except (ValueError, TypeError) as cls_err:
            logger.debug("Invalid classification label in %s: %s", label_path, cls_err)
            return None

    @staticmethod
    def _folder_class(sample: DirectorySample) -> int:
        """Class index from the parent folder name (a number or a ``classes.txt`` entry)."""
        parent_dir = os.path.dirname(sample.paths.fname)
        if not parent_dir:
            return 0
        class_candidate = os.path.basename(parent_dir)
        if class_candidate.isdigit():
            return int(class_candidate)
        classes_file = os.path.join(sample.paths.ds_path, "classes.txt")
        try:
            c_lines = [line.strip() for line in read_lines(classes_file) if line.strip()]
        except OSError as cf_err:
            logger.debug("Failed reading classes file %s: %s", classes_file, cf_err)
            return 0
        return c_lines.index(class_candidate) if class_candidate in c_lines else 0

    def segmentation_item(self, sample: DirectorySample):
        """Segmentation items: the class mask is the target."""
        return sample.img_tensor, self.load_mask(sample), "segmentation"

    def face_detection_item(self, sample: DirectorySample):
        """Face detection: confidence plus box and landmark coordinates (15 values)."""
        target = torch.zeros(15, dtype=torch.float32)
        label_path = self.label_path(sample)
        try:
            text = read_text(label_path)
            parts = text.split() if text is not None else []
            if len(parts) >= 15:
                target[0] = 1.0  # confidence
                for i in range(14):
                    target[i + 1] = float(parts[i + 1])
        except (ValueError, TypeError) as fd_err:
            logger.debug("Invalid face detection label in %s: %s", label_path, fd_err)
        return sample.img_tensor, target, "face_detection"

    def task_item(self, sample: DirectorySample):
        """Dispatch ``sample`` to its task type's builder."""
        if self.spec.task_type in RESTORATION_TASKS:
            return self.restoration_item(sample)
        builders = {
            "parameter_prediction": self.parameter_item,
            "quality": self.quality_item,
            "classification": self.classification_item,
            "segmentation": self.segmentation_item,
            "face_detection": self.face_detection_item,
        }
        builder = builders.get(self.spec.task_type)
        if builder is None:
            return sample.img_tensor, torch.zeros(1), self.spec.task_type
        return builder(sample)
