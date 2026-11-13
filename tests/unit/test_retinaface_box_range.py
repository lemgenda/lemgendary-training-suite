"""RetinaFace after the SFHQ label rebuild: tight one-face boxes are fully covered.

The LemGendizedRetinaFace labels now carry one tight WIDER-style box per image (median
w 0.55 x h 0.65 of the image; the old boxes were about 0.86 x 0.97). The suite's RetinaFace
(``models.detection.RetinaFaceMobileNet``) has no prior boxes: one globally pooled head
regresses a single normalized ``cx cy w h`` box per image, so there are no ``min_sizes``,
``steps`` or matching IoU thresholds, and the training resolution does not limit the box
sizes it can match. These tests compute that coverage for synthetic boxes in the new and the
old size range and guard the assumption: when the registry or the model gains priors, the
guard fails and a real anchor coverage check is due.
"""

from __future__ import annotations

import random

import pytest
import torch

from data.sample_labels import FACE_BOX_SLICE, face_target_from_json
from models.detection import RetinaFaceMobileNet
from training.losses import _face_detection_loss
from training.notebooks.registry import load_registry

_MATCH_IOU = 0.35
"""Reference RetinaFace positive threshold; the suite itself matches no anchors."""
_NEW_RANGE = ((0.45, 0.65), (0.55, 0.75))
_OLD_RANGE = ((0.80, 0.92), (0.92, 1.0))
_PRIOR_KEYS = ("min_sizes", "steps", "anchors", "priors", "match_iou", "variance")


def _synthetic_boxes(ranges: tuple[tuple[float, float], ...], count: int) -> list[list[float]]:
    """``count`` normalized ``cx cy w h`` boxes inside the image, sizes drawn from ``ranges``."""
    rng = random.Random(0)
    (w_lo, w_hi), (h_lo, h_hi) = ranges
    boxes = []
    for _ in range(count):
        w, h = rng.uniform(w_lo, w_hi), rng.uniform(h_lo, h_hi)
        boxes.append([rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2), w, h])
    return boxes


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise IoU of two ``[N, 4]`` ``cx cy w h`` tensors."""
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp(min=0).prod(dim=1)
    return inter / (a[:, 2:].prod(dim=1) + b[:, 2:].prod(dim=1) - inter)


@pytest.mark.parametrize("ranges", [_NEW_RANGE, _OLD_RANGE], ids=["rebuilt", "old"])
def test_single_box_head_covers_every_face(ranges: tuple[tuple[float, float], ...]) -> None:
    """Every synthetic box is a positive target the box head is trained on (coverage 1.0)."""
    boxes = _synthetic_boxes(ranges, 256)
    rows = []
    for box in boxes:
        row = face_target_from_json({"objects": [{"bbox_cxcywh_normalized": box}]})
        assert row is not None
        rows.append(row)
    targets = torch.stack(rows)
    encoded = targets[:, FACE_BOX_SLICE]
    assert torch.allclose(encoded, torch.tensor(boxes))
    assert bool((targets[:, 0] == 1.0).all())

    coverage = float((_iou(encoded, torch.tensor(boxes)) > _MATCH_IOU).float().mean())
    assert coverage == 1.0

    batch = len(boxes)
    exact = (encoded.clone(), torch.ones(batch, 1), torch.zeros(batch, 10))
    shifted = (encoded + 0.05, torch.ones(batch, 1), torch.zeros(batch, 10))
    assert _face_detection_loss(shifted, targets) > _face_detection_loss(exact, targets)


def test_retinaface_has_no_priors_to_size() -> None:
    """Guard: neither the registry entry nor the model defines prior boxes."""
    entry = load_registry()["retinaface"]
    assert not [key for key in _PRIOR_KEYS if key in entry or key in entry.get("kwargs", {})]
    assert RetinaFaceMobileNet.head_param_prefixes == (
        "bbox_head.",
        "conf_head.",
        "landmark_head.",
    )
    assert not [name for name in vars(RetinaFaceMobileNet) if "prior" in name or "anchor" in name]
