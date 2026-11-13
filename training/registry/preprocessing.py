"""The image preprocessing of a model (single reader of the registry ``preprocessing`` key).

Every image model reads RGB pixels scaled to ``[0, 1]``; some then subtract a per-channel
mean and divide by a standard deviation. An entry may declare it::

    preprocessing:
      normalization: unit      # or imagenet

Without the key the task type decides (:data:`TASK_NORMALIZATION`): ``quality`` models
(NIMA) are ImageNet-normalized, every other task trains on ``[0, 1]`` pixels. Training and
validation transforms, the export metadata, the model card and the generated inference
cells all read :func:`image_preprocessing`, so they cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PREPROCESSING_KEY = "preprocessing"
NORMALIZATION_KEY = "normalization"


@dataclass(frozen=True)
class ImagePreprocessing:
    """Pixel scaling then ``(x - mean) / std`` per RGB channel."""

    name: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    scale: float = 1.0 / 255.0

    @property
    def normalizes(self) -> bool:
        """True when a mean/std step follows the ``[0, 1]`` scaling."""
        return self.mean != (0.0, 0.0, 0.0) or self.std != (1.0, 1.0, 1.0)

    def describe(self) -> str:
        """One sentence for documentation (model card)."""
        if not self.normalizes:
            return "RGB pixels scaled to [0, 1] (x / 255), no mean/std normalization."
        return (
            f"RGB pixels scaled to [0, 1] (x / 255), then (x - mean) / std with "
            f"{self.name} mean {list(self.mean)} and std {list(self.std)}."
        )

    def metadata(self) -> dict[str, str]:
        """String metadata of the preprocessing (export files)."""
        return {
            "normalization": self.name,
            "pixel_scale": "1/255",
            "mean": ",".join(f"{value:g}" for value in self.mean),
            "std": ",".join(f"{value:g}" for value in self.std),
        }


UNIT = ImagePreprocessing("unit", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
IMAGENET = ImagePreprocessing("imagenet", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
PREPROCESSINGS: dict[str, ImagePreprocessing] = {p.name: p for p in (UNIT, IMAGENET)}
TASK_NORMALIZATION: dict[str, str] = {"quality": IMAGENET.name}
"""Normalization of a task type without a registry ``preprocessing`` key (else ``unit``)."""


class PreprocessingError(ValueError):
    """A registry ``preprocessing`` value that is malformed or names no known normalization."""


def task_type_of(model_info: Mapping[str, Any]) -> str:
    """The entry's primary ``dataset_type`` (the first of a list; ``restoration`` default)."""
    raw = model_info.get("dataset_type", "restoration")
    return str(raw[0] if isinstance(raw, list) and raw else raw)


def image_preprocessing(model_info: Mapping[str, Any] | None) -> ImagePreprocessing:
    """The entry's validated image preprocessing.

    ``PreprocessingError`` for a non-mapping ``preprocessing`` or an unknown normalization.
    """
    info = model_info or {}
    raw = info.get(PREPROCESSING_KEY)
    if raw is None:
        return PREPROCESSINGS[TASK_NORMALIZATION.get(task_type_of(info), UNIT.name)]
    if not isinstance(raw, Mapping):
        raise PreprocessingError(f"preprocessing must be a mapping, got {raw!r}")
    name = raw.get(NORMALIZATION_KEY)
    if name not in PREPROCESSINGS:
        raise PreprocessingError(
            f"preprocessing.normalization must be one of {sorted(PREPROCESSINGS)}, got {name!r}"
        )
    return PREPROCESSINGS[name]
