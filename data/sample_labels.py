"""Label parsing shared by directory-backed and container-backed samples.

Every function turns one raw label (a label file's text, a JSON sidecar or an already
decoded value) into the target tensor a task expects, so both sample sources produce the
same targets:

* quality: a 10-bin NIMA score distribution,
* face detection: ``[conf, cx, cy, w, h, 10 landmark coordinates]`` (15 values),
* FFA-Net: up to 50 five-value YOLO boxes.

Landmark coordinates a label does not carry are set to :data:`LANDMARK_ABSENT`; the face
detection loss masks them out instead of training towards made-up points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

NIMA_BINS = 10
"""Bins of a NIMA score distribution."""

FACE_TARGET_LEN = 15
"""Face detection target length: confidence, 4 box values and 10 landmark coordinates."""

FACE_BOX_SLICE = slice(1, 5)
FACE_LANDMARK_SLICE = slice(5, FACE_TARGET_LEN)

LANDMARK_ABSENT = -1.0
"""Landmark value of a face whose label has no landmarks (normalised coordinates are >= 0)."""

MAX_FFANET_BOXES = 50
"""Boxes kept per FFA-Net sample (zero-padded)."""

_NIMA_KEYS = ("distribution", "scores")
_LANDMARK_KEYS = ("landmarks_normalized", "landmarks", "keypoints")
_AUTHENTIC_THRESHOLD = 5.0
# 'real' peaks at bin 10 (class 1); 'shutterstock' / synthetic peaks at bin 1 (class 0).
_AUTHENTIC_DISTRIBUTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9)
_SYNTHETIC_DISTRIBUTION = (0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _floats(values: Any) -> list[float] | None:
    """``values`` as floats: a whitespace-separated string or a numeric sequence."""
    if isinstance(values, str):
        values = values.split()
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _nima_distribution(label: Mapping[str, Any]) -> Any:
    """The score list of a NIMA JSON mapping (``nima_scores`` nested or flat)."""
    nested = label.get("nima_scores")
    for source in (nested, label):
        if isinstance(source, Mapping):
            for key in _NIMA_KEYS:
                if source.get(key) is not None:
                    return source[key]
    return None


def quality_scores(label: Any) -> torch.Tensor | None:
    """The 10-bin score distribution of ``label``; ``None`` when it carries none.

    Accepts a numeric sequence, a label file's whitespace-separated text or a NIMA JSON
    mapping (``{"nima_scores": {"distribution": [...]}}`` or a flat ``distribution`` /
    ``scores`` key). Short distributions are zero-padded, long ones truncated.
    """
    if isinstance(label, Mapping):
        label = _nima_distribution(label)
    scores = _floats(label)
    if not scores:
        return None
    scores = (scores + [0.0] * NIMA_BINS)[:NIMA_BINS]
    return torch.tensor(scores, dtype=torch.float32)


def authenticity_distribution(mean_score: float) -> torch.Tensor:
    """Binary authenticity target from a NIMA mean score (above 5 means authentic)."""
    peaks = (
        _AUTHENTIC_DISTRIBUTION if mean_score > _AUTHENTIC_THRESHOLD else _SYNTHETIC_DISTRIBUTION
    )
    return torch.tensor(peaks, dtype=torch.float32)


def _face_target(box: Sequence[float], landmarks: Sequence[float] | None) -> torch.Tensor:
    target = torch.full((FACE_TARGET_LEN,), LANDMARK_ABSENT, dtype=torch.float32)
    target[0] = 1.0
    target[FACE_BOX_SLICE] = torch.tensor(list(box[:4]), dtype=torch.float32)
    if landmarks is not None and len(landmarks) >= 10:
        target[FACE_LANDMARK_SLICE] = torch.tensor(list(landmarks[:10]), dtype=torch.float32)
    return target


def empty_face_target() -> torch.Tensor:
    """Target of an image without a labelled face (confidence 0, everything else 0)."""
    return torch.zeros(FACE_TARGET_LEN, dtype=torch.float32)


def face_target_from_text(text: str | None) -> torch.Tensor | None:
    """Face target from the first YOLO line ``cls cx cy w h [10 landmark values]``.

    ``None`` when no line has a class and a box.
    """
    for line in (text or "").splitlines():
        values = _floats(line)
        if values is not None and len(values) >= 5:
            landmarks = values[5:15] if len(values) >= 15 else None
            return _face_target(values[1:5], landmarks)
    return None


def face_target_from_json(meta: Any) -> torch.Tensor | None:
    """Face target from a JSON sidecar's first object with ``bbox_cxcywh_normalized``."""
    objects = meta.get("objects") if isinstance(meta, Mapping) else None
    if not isinstance(objects, Sequence):
        return None
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        box = _floats(obj.get("bbox_cxcywh_normalized"))
        if box is None or len(box) < 4:
            continue
        landmarks = None
        for key in _LANDMARK_KEYS:
            landmarks = _flat_floats(obj.get(key))
            if landmarks is not None:
                break
        return _face_target(box, landmarks)
    return None


def _flat_floats(values: Any) -> list[float] | None:
    """Floats of a flat sequence or of ``[[x, y], ...]`` point pairs."""
    if (
        isinstance(values, Sequence)
        and values
        and all(isinstance(v, Sequence) and not isinstance(v, str) for v in values)
    ):
        return _floats([coord for point in values for coord in point])
    return _floats(values)


def ffanet_boxes(text: str | None) -> torch.Tensor:
    """Up to 50 five-value YOLO boxes from label text, zero-padded to ``[50, 5]``."""
    boxes: list[list[float]] = []
    for line in (text or "").splitlines():
        values = _floats(line)
        if values is not None and len(values) >= 5:
            boxes.append(values[:5])
    tensor = torch.zeros((MAX_FFANET_BOXES, 5), dtype=torch.float32)
    if boxes:
        count = min(len(boxes), MAX_FFANET_BOXES)
        tensor[:count] = torch.tensor(boxes[:count], dtype=torch.float32)
    return tensor
