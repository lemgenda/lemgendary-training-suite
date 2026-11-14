"""RetinaFace validation metrics: box IoU, AP over images and landmark NME (measured only).

The old detection score was ``1 - |sigmoid(conf) - target|`` on a confidence the model had
already passed through a sigmoid (capped near 0.73), and ``map_*`` were fixed multiples of
it. These tests pin the measured replacements: a perfect prediction scores IoU 1, AP 1 and
NME 0; shifted boxes and landmarks lower all three.
"""

from __future__ import annotations

import pytest
import torch

from data.sample_labels import empty_face_target, face_target_from_json
from training.governance.metrics import FACE_DETECTION_METRICS, is_lower_better
from training.governance.quality_score import (
    FACE_DETECTION_SCHEME,
    WEIGHTED_SCHEME,
    anchor_is_comparable,
    quality_scheme,
    score_metrics,
    target_quality_score,
)
from training.notebooks.registry import load_registry
from training.server.routes.gui_model_stats import METRIC_REGISTRY
from training.telemetry import FACE_DETECTION_LAYOUT, EpochLosses, EpochRow, TelemetryEngine
from training.training.face_detection_metrics import (
    average_precision,
    box_iou_cxcywh,
    face_detection_results,
    landmark_nme,
)
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import ValBatch, update_face_detection

_BOXES = [
    [0.50, 0.50, 0.55, 0.65],
    [0.45, 0.52, 0.50, 0.60],
    [0.55, 0.48, 0.60, 0.70],
    [0.50, 0.55, 0.45, 0.62],
]
_LANDMARKS = [0.38, 0.42, 0.62, 0.42, 0.50, 0.55, 0.40, 0.68, 0.60, 0.68]
_SOTA = {"box_iou": 0.72, "ap50": 0.95, "ap50_95": 0.6, "landmark_nme": 0.05}


def _targets() -> torch.Tensor:
    rows = []
    for box in _BOXES:
        row = face_target_from_json(
            {"objects": [{"bbox_cxcywh_normalized": box, "landmarks_normalized": _LANDMARKS}]}
        )
        assert row is not None
        rows.append(row)
    return torch.stack(rows)


def _measure(boxes: torch.Tensor, conf: torch.Tensor, landmarks: torch.Tensor, targets=None):
    targets = _targets() if targets is None else targets
    acc = ValAccumulators()
    batch = ValBatch(
        preds=(boxes, conf, landmarks),
        targets=targets,
        sota_cfg=_SOTA,
        dataset_type="face_detection",
        device=torch.device("cpu"),
        governor=None,
    )
    update_face_detection(acc, batch)
    return face_detection_results(acc.faces), acc


def _perfect() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = _targets()
    return targets[:, 1:5].clone(), torch.ones(len(_BOXES), 1), targets[:, 5:].clone()


def test_perfect_prediction_scores_iou_one_ap_one_nme_zero() -> None:
    """Exact boxes and landmarks: IoU 1, AP50 1, AP50:95 1 and NME 0."""
    boxes, conf, landmarks = _perfect()
    results, _ = _measure(boxes, conf, landmarks)
    assert set(results) == set(FACE_DETECTION_METRICS)
    assert results["box_iou"] == pytest.approx(1.0)
    assert results["ap50"] == pytest.approx(1.0)
    assert results["ap50_95"] == pytest.approx(1.0)
    assert results["landmark_nme"] == pytest.approx(0.0, abs=1e-7)


def test_low_confidence_is_not_squashed_again() -> None:
    """The model returns probabilities: a uniform 0.6 still ranks perfect boxes at AP 1."""
    boxes, _, landmarks = _perfect()
    results, _ = _measure(boxes, torch.full((len(_BOXES), 1), 0.6), landmarks)
    assert results["ap50"] == pytest.approx(1.0)
    assert results["ap50_95"] == pytest.approx(1.0)


def test_small_shift_lowers_iou_ap50_95_and_raises_nme() -> None:
    """Boxes and landmarks moved by 0.05: every metric gets worse, AP50 still holds."""
    boxes, conf, landmarks = _perfect()
    perfect, _ = _measure(boxes, conf, landmarks)
    shifted, _ = _measure(boxes + torch.tensor([0.05, 0.05, 0.0, 0.0]), conf, landmarks + 0.05)
    assert shifted["box_iou"] < perfect["box_iou"]
    assert shifted["ap50_95"] < perfect["ap50_95"]
    assert shifted["landmark_nme"] > perfect["landmark_nme"]
    assert shifted["ap50"] == pytest.approx(1.0)


def test_large_shift_lowers_ap50() -> None:
    """Boxes moved by half their size fall below IoU 0.5 and stop counting at AP50."""
    boxes, conf, landmarks = _perfect()
    shifted, _ = _measure(boxes + torch.tensor([0.3, 0.3, 0.0, 0.0]), conf, landmarks)
    assert shifted["box_iou"] < 0.5
    assert shifted["ap50"] < 1.0
    assert shifted["ap50_95"] < 1.0


def test_ap_ranks_by_confidence() -> None:
    """A wrong box ranked first costs more AP than the same wrong box ranked last."""
    ious = [1.0, 1.0, 1.0, 0.1]
    faces = [True, True, True, True]
    wrong_last = average_precision([0.9, 0.8, 0.7, 0.1], ious, faces, 0.5)
    wrong_first = average_precision([0.5, 0.4, 0.3, 0.9], ious, faces, 0.5)
    assert wrong_last == pytest.approx(0.75)
    assert wrong_first is not None and wrong_last is not None
    assert wrong_first < wrong_last


def test_image_without_face_is_a_false_positive() -> None:
    """A confident detection on a face-less image lowers AP; it adds no IoU sample."""
    targets = torch.stack([_targets()[0], empty_face_target()])
    boxes = torch.stack([targets[0, 1:5], torch.tensor([0.5, 0.5, 0.4, 0.4])])
    landmarks = torch.stack([targets[0, 5:], torch.zeros(10)])
    results, _ = _measure(boxes, torch.tensor([[0.5], [0.9]]), landmarks, targets)
    assert results["box_iou"] == pytest.approx(1.0)
    assert results["ap50"] == pytest.approx(0.5)


def test_absent_landmarks_are_left_out_of_nme() -> None:
    """A box-only label gives no NME sample; it is not scored against made-up points."""
    row = face_target_from_json({"objects": [{"bbox_cxcywh_normalized": _BOXES[0]}]})
    assert row is not None
    target = row.unsqueeze(0)
    assert landmark_nme(torch.zeros(1, 10), target[:, 5:]).numel() == 0
    results, _ = _measure(target[:, 1:5].clone(), torch.ones(1, 1), torch.zeros(1, 10), target)
    assert "landmark_nme" not in results
    assert results["box_iou"] == pytest.approx(1.0)


def test_negative_box_size_has_zero_iou() -> None:
    """The unbounded box head may emit a negative width: that box is empty, IoU 0."""
    iou = box_iou_cxcywh(torch.tensor([[0.5, 0.5, -0.2, 0.4]]), torch.tensor([_BOXES[0]]))
    assert float(iou[0]) == 0.0


def test_other_models_record_nothing() -> None:
    """Models without face targets in ``sota_targets`` skip the face updater."""
    boxes, conf, landmarks = _perfect()
    acc = ValAccumulators()
    batch = ValBatch(
        preds=(boxes, conf, landmarks),
        targets=_targets(),
        sota_cfg={"map50": 0.54},
        dataset_type="detection",
        device=torch.device("cpu"),
        governor=None,
    )
    update_face_detection(acc, batch)
    assert not acc.faces.face_ious
    assert not face_detection_results(acc.faces)


def test_progress_snapshot_round_trip() -> None:
    """The per-image face totals survive an intra-epoch progress checkpoint."""
    boxes, conf, landmarks = _perfect()
    results, acc = _measure(boxes, conf, landmarks)
    restored = ValAccumulators()
    restored.restore(acc.snapshot())
    assert face_detection_results(restored.faces) == results


def test_registry_targets_name_the_measured_metrics() -> None:
    """RetinaFace targets are the measured keys, calibrated, with no ``map_*``."""
    entry = load_registry()["retinaface"]
    assert tuple(entry["sota_targets"]) == FACE_DETECTION_METRICS
    assert entry["sota_targets_provisional"] is False
    assert is_lower_better("landmark_nme")
    for key in FACE_DETECTION_METRICS:
        assert key in METRIC_REGISTRY


def test_score_uses_measured_metrics_and_its_own_scheme() -> None:
    """The score rises with the metrics; old untagged scores are not reused as a best."""
    targets = load_registry()["retinaface"]["sota_targets"]
    good = {"box_iou": 0.9, "ap50": 1.0, "ap50_95": 0.7, "landmark_nme": 0.05}
    poor = {"box_iou": 0.4, "ap50": 0.3, "ap50_95": 0.1, "landmark_nme": 0.4}
    assert score_metrics(good, targets, "face_detection") > score_metrics(
        poor, targets, "face_detection"
    )
    assert target_quality_score(targets, "face_detection") == pytest.approx(
        100.0 * (0.72 + 0.95 + 0.6 + (1.0 - 0.05))
    )
    assert quality_scheme("face_detection") == FACE_DETECTION_SCHEME
    assert not anchor_is_comparable(None, "face_detection")
    assert not anchor_is_comparable(WEIGHTED_SCHEME, "face_detection")


def test_metrics_csv_face_layout(tmp_path) -> None:
    """RetinaFace rows carry the measured metric columns; unmeasured cells stay empty."""
    engine = TelemetryEngine(str(tmp_path), task_type="face_detection")
    engine.write_epoch_row(
        EpochRow(
            epoch=1,
            losses=EpochLosses(train_loss=0.5, val_loss=0.4, lr=1e-4),
            metrics={"box_iou": 0.75, "ap50": 0.9, "ap50_95": 0.5},
            quality_score=215.0,
            governor_state=None,
            stress=0.0,
        )
    )
    header, row = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert header == FACE_DETECTION_LAYOUT.header.strip()
    cells = dict(zip(header.split(","), row.split(","), strict=True))
    assert cells["Box_IoU"] == "0.7500"
    assert cells["AP50"] == "0.9000"
    assert cells["AP50_95"] == "0.5000"
    assert cells["Landmark_NME"] == ""
