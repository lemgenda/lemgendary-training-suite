"""All-zero or missing face landmarks are absent, not points at the image corner.

YOLO text and JSON sidecar labels decode the same way (datasets ``face_landmarks`` strategy).
"""

from __future__ import annotations

import pytest
import torch

from data.sample_labels import (
    FACE_LANDMARK_SLICE,
    LANDMARK_ABSENT,
    face_target_from_json,
    face_target_from_text,
)

BOX = [0.5, 0.4, 0.2, 0.3]
POINTS = [0.45, 0.35, 0.55, 0.35, 0.5, 0.4, 0.46, 0.46, 0.54, 0.46]
ZEROS = [0.0] * 10


def _line(values: list[float]) -> str:
    return " ".join(str(v) for v in [0, *values])


def _json(landmarks: object | None) -> dict[str, object]:
    obj: dict[str, object] = {"bbox_cxcywh_normalized": BOX}
    if landmarks is not None:
        obj["landmarks_normalized"] = landmarks
    return {"objects": [obj]}


def _absent(target: torch.Tensor | None) -> bool:
    assert target is not None
    assert target[0] == 1.0
    assert torch.allclose(target[1:5], torch.tensor(BOX))
    return bool(torch.all(target[FACE_LANDMARK_SLICE] == LANDMARK_ABSENT))


@pytest.mark.parametrize("text", [_line([*BOX, *ZEROS]), _line(BOX)], ids=["all-zero", "box-only"])
def test_text_labels_without_real_landmarks(text: str) -> None:
    """A YOLO line with 10 zero landmarks or only a box has absent landmarks."""
    assert _absent(face_target_from_text(text))


@pytest.mark.parametrize(
    "landmarks",
    [ZEROS, [[0.0, 0.0]] * 5, None],
    ids=["all-zero-flat", "all-zero-points", "missing"],
)
def test_json_labels_without_real_landmarks(landmarks: object | None) -> None:
    """A JSON object with all-zero or no landmarks has absent landmarks."""
    assert _absent(face_target_from_json(_json(landmarks)))


def test_real_landmarks_are_kept_in_both_formats() -> None:
    """Real points (one coordinate may be zero) are decoded unchanged."""
    points = [0.0, *POINTS[1:]]
    for target in (
        face_target_from_text(_line([*BOX, *points])),
        face_target_from_json(_json(points)),
    ):
        assert target is not None
        assert torch.allclose(target[FACE_LANDMARK_SLICE], torch.tensor(points))
