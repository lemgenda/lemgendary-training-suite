"""Single-box face detection metrics of RetinaFace (one face per image, no anchors).

``models.detection.RetinaFaceMobileNet`` regresses one normalized ``cx cy w h`` box, one
face probability (the sigmoid is applied inside ``forward``) and five landmarks (``x y``
interleaved, image-left eye first) per image. The LemGendizedRetinaFace labels hold
exactly one face per image. The validation metrics measure that output directly:

* ``box_iou``: mean IoU of the predicted box with the ground-truth box over the images
  that hold a face.
* ``ap50`` / ``ap50_95``: single-class average precision over images. Every image gives
  one detection, ranked by its confidence; it is a true positive when the image holds a
  face and the box IoU reaches the threshold, else a false positive. AP is the area under
  the precision envelope (all-point interpolation, PASCAL VOC 2010 onwards); ``ap50_95``
  averages AP over the COCO IoU thresholds 0.50:0.05:0.95.
* ``landmark_nme``: mean distance of the five predicted points from the labelled ones,
  divided by the inter-ocular distance (points 0 and 1), averaged over the images whose
  label carries landmarks.

These are not WIDER FACE easy / medium / hard mAP: the model detects one box per image
and the validation set is not WIDER FACE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data.sample_labels import FACE_BOX_SLICE, FACE_LANDMARK_SLICE, LANDMARK_ABSENT
from training.governance.metrics import FACE_DETECTION_METRICS
from training.training.val_accumulators import FaceDetectionTotals

__all__ = [
    "FACE_DETECTION_METRICS",
    "FacePrediction",
    "average_precision",
    "box_iou_cxcywh",
    "face_detection_results",
    "landmark_nme",
    "record_face_batch",
]

AP50_THRESHOLD = 0.5
AP50_95_THRESHOLDS = tuple(float(t) for t in np.linspace(0.5, 0.95, 10))
"""COCO IoU thresholds 0.50, 0.55, ..., 0.95."""

_EPS = 1e-9
_LANDMARK_POINTS = 5


def box_iou_cxcywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Row-wise IoU of two ``[N, 4]`` normalized ``cx cy w h`` box tensors.

    A negative predicted width or height (the box head is an unbounded linear layer) is
    an empty box with IoU 0.
    """
    pred_wh = pred[:, 2:].clamp(min=0.0)
    target_wh = target[:, 2:].clamp(min=0.0)
    pred_lo, pred_hi = pred[:, :2] - pred_wh / 2, pred[:, :2] + pred_wh / 2
    target_lo, target_hi = target[:, :2] - target_wh / 2, target[:, :2] + target_wh / 2
    overlap = (torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo)).clamp(min=0.0)
    inter = overlap.prod(dim=1)
    union = pred_wh.prod(dim=1) + target_wh.prod(dim=1) - inter
    return inter / union.clamp(min=_EPS)


def landmark_nme(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image NME of ``[N, 10]`` landmarks, for rows with labelled points only.

    Rows whose label marks the landmarks absent (:data:`LANDMARK_ABSENT`) or whose two
    eye points coincide are left out; the result has one value per remaining row.
    """
    pred_pts = pred.reshape(-1, _LANDMARK_POINTS, 2)
    target_pts = target.reshape(-1, _LANDMARK_POINTS, 2)
    labelled = (target > LANDMARK_ABSENT / 2.0).all(dim=1)
    inter_ocular = torch.linalg.vector_norm(target_pts[:, 0] - target_pts[:, 1], dim=-1)
    usable = labelled & (inter_ocular > _EPS)
    errors = torch.linalg.vector_norm(pred_pts - target_pts, dim=-1).mean(dim=1)
    return errors[usable] / inter_ocular[usable]


def average_precision(
    confidences: list[float], ious: list[float], has_face: list[bool], threshold: float
) -> float | None:
    """All-point interpolated AP of one detection per image; ``None`` without a face."""
    positives = sum(1 for present in has_face if present)
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    hits = np.array(
        [has_face[i] and ious[i] >= threshold for i in order.tolist()], dtype=np.float64
    )
    true_pos = np.cumsum(hits)
    recall = true_pos / positives
    precision = true_pos / np.arange(1, len(hits) + 1, dtype=np.float64)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    steps = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


@dataclass(frozen=True)
class FacePrediction:
    """One batch of RetinaFace outputs: boxes ``[B, 4]``, probabilities, landmarks ``[B, 10]``."""

    boxes: torch.Tensor
    confidence: torch.Tensor
    landmarks: torch.Tensor


def record_face_batch(
    totals: FaceDetectionTotals, pred: FacePrediction, target: torch.Tensor
) -> None:
    """Add one batch to ``totals``; ``target`` rows are ``[conf, cx, cy, w, h, 10 landmarks]``.

    The confidence is used as the model returns it: a probability, never passed through
    another sigmoid.
    """
    target = target.detach().float()
    face_rows = target[:, 0] > 0.5
    ious = box_iou_cxcywh(pred.boxes.detach().float(), target[:, FACE_BOX_SLICE])
    confidences = pred.confidence.detach().float().reshape(-1)
    nme = landmark_nme(
        pred.landmarks.detach().float()[face_rows], target[:, FACE_LANDMARK_SLICE][face_rows]
    )
    if not confidences.numel() == ious.numel() == face_rows.numel():
        raise ValueError(
            f"Face batch sizes differ: {confidences.numel()} confidences, "
            f"{ious.numel()} boxes, {face_rows.numel()} targets"
        )
    totals.face_confidences.extend(confidences.cpu().tolist())
    totals.face_ious.extend(ious.cpu().tolist())
    totals.face_present.extend(bool(v) for v in face_rows.cpu().tolist())
    totals.face_nme.extend(nme.cpu().tolist())


def face_detection_results(totals: FaceDetectionTotals) -> dict[str, float]:
    """The measured face detection metrics of a pass; empty without a labelled face."""
    present = totals.face_present
    face_ious = [iou for iou, face in zip(totals.face_ious, present) if face]
    if not face_ious:
        return {}
    results = {"box_iou": float(np.mean(face_ious))}
    ap50 = average_precision(totals.face_confidences, totals.face_ious, present, AP50_THRESHOLD)
    per_threshold = [
        average_precision(totals.face_confidences, totals.face_ious, present, t)
        for t in AP50_95_THRESHOLDS
    ]
    if ap50 is not None:
        results["ap50"] = ap50
    if all(ap is not None for ap in per_threshold):
        results["ap50_95"] = float(np.mean([ap for ap in per_threshold if ap is not None]))
    if totals.face_nme:
        results["landmark_nme"] = float(np.mean(totals.face_nme))
    return results
