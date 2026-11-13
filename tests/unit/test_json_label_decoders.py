"""Shard label decoders read only the sample's JSON document (R4-J2).

Boxes come from ``objects`` (``category_id``, or ``class_name`` mapped through the
manifold's ``dataset_info.yaml`` ``classes``), faces only from JSON, classification from
``class_id`` / ``class_name``, captions from ``caption``. A missing or malformed document
is reported, never turned into a default label (an unlabeled classification sample still
gets ``IGNORE_TARGET``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from data.dataset import MultiTaskDataset
from data.json_labels import (
    LabelDocumentError,
    caption,
    class_label,
    object_boxes,
    sample_document,
    yolo_rows,
)
from data.manifold_classes import ManifoldClassesError, manifold_classes
from data.sample_labels import FACE_TARGET_LEN, MAX_FFANET_BOXES
from tests.fixtures.webdataset import json_bytes, single_shard_config, webp_bytes
from training.ignored_targets import IGNORE_TARGET

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
_IMAGE = webp_bytes((40, 90, 160), size=(24, 24), lossless=True)
_TARGET = webp_bytes((200, 210, 220), size=(24, 24), lossless=True)
BOX = [0.5, 0.4, 0.2, 0.3]
KEY = "s_0001"


def _obj(**fields: Any) -> dict[str, Any]:
    return {"bbox_cxcywh_normalized": BOX, **fields}


# --- data.json_labels ---------------------------------------------------------------


def test_yolo_rows_map_class_names_through_the_manifold_classes() -> None:
    """category_id is kept; class_name maps to its dataset_info classes index."""
    doc = {"objects": [_obj(category_id=2), _obj(class_name="car")]}
    assert yolo_rows(doc, ("person", "bike", "car")) == [(2, *BOX), (2, *BOX)]


def test_yolo_rows_without_objects_and_with_none() -> None:
    """No objects key gives None; an empty objects list is an image without objects."""
    assert yolo_rows({"caption": "x"}) is None
    assert yolo_rows({"object_count": 0, "objects": []}) == []


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"objects": [_obj(class_name="truck")]}, "not a manifold class"),
        ({"objects": [_obj()]}, "neither category_id nor class_name"),
        ({"objects": [_obj(category_id=-1)]}, "non-negative"),
        (
            {"objects": [{"category_id": 0, "bbox_cxcywh_normalized": [0.5, 2.0, 0.1, 0.1]}]},
            "4 numbers in 0..1",
        ),
        ({"objects": {"category_id": 0}}, "not a list of objects"),
    ],
)
def test_yolo_rows_reject_malformed_objects(doc: dict[str, Any], match: str) -> None:
    """Schema breaks raise LabelDocumentError instead of becoming empty boxes."""
    with pytest.raises(LabelDocumentError, match=match):
        yolo_rows(doc, ("person", "car"))


def test_object_boxes_ignore_classes() -> None:
    """Face boxes are read without a class lookup."""
    assert object_boxes({"objects": [_obj()]}) == [BOX]
    assert object_boxes({"face_count": 0}) is None


def test_class_label_and_caption() -> None:
    """Classification labels and captions come from their JSON keys."""
    assert class_label({"class_id": 0}) == 0
    assert class_label({"class_name": "cat"}) == "cat"
    assert class_label({"objects": []}) is None
    assert caption({"caption": " a red house "}) == "a red house"
    assert caption({"class_id": 1}) is None
    with pytest.raises(LabelDocumentError, match="not a string"):
        caption({"caption": 3})


def test_sample_document_reports_unreadable_json() -> None:
    """An unreadable or non-object JSON label raises; a missing one is None."""
    assert sample_document({}, "k") is None
    with pytest.raises(LabelDocumentError, match="unreadable JSON label"):
        sample_document({"json_error": "Expecting value"}, "k")
    with pytest.raises(LabelDocumentError, match="not an object"):
        sample_document({"json": [1, 2]}, "k")


# --- data.manifold_classes ----------------------------------------------------------


def test_manifold_classes(tmp_path: Path) -> None:
    """dataset_info.yaml classes: absent, read in id order, or rejected when malformed."""
    assert not manifold_classes(tmp_path)
    (tmp_path / "dataset_info.yaml").write_text("classes: [cat, ' dog ']\n", encoding="utf-8")
    assert manifold_classes(tmp_path) == ("cat", "dog")
    (tmp_path / "dataset_info.yaml").write_text("classes: [cat, 3]\n", encoding="utf-8")
    with pytest.raises(ManifoldClassesError):
        manifold_classes(tmp_path)


# --- container decoders -------------------------------------------------------------


def _dataset(
    tmp_path: Path, model_key: str, members: dict[str, bytes], classes: list[str] | None = None
) -> MultiTaskDataset:
    name = _REGISTRY[model_key]["datasets"][0]
    config = single_shard_config(tmp_path / "datasets", name, members)
    if classes is not None:
        info = tmp_path / "datasets" / name / "dataset_info.yaml"
        info.write_text(yaml.safe_dump({"classes": classes}), encoding="utf-8")
    return MultiTaskDataset(config, model_key=model_key, is_train=False, env="local")


def _boxes(dataset: MultiTaskDataset) -> torch.Tensor:
    """FFA-Net boxes of the dataset's first sample."""
    target = dataset[0][1]
    assert isinstance(target, dict)
    return target["bboxes"]


def _tensor_target(dataset: MultiTaskDataset) -> torch.Tensor:
    """Tensor target of the dataset's first sample."""
    target = dataset[0][1]
    assert isinstance(target, torch.Tensor)
    return target


def _members(doc: dict[str, Any] | None = None, raw: bytes | None = None) -> dict[str, bytes]:
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.target.webp": _TARGET}
    if doc is not None:
        members[f"{KEY}.json"] = json_bytes(doc)
    if raw is not None:
        members[f"{KEY}.json"] = raw
    return members


def test_ffanet_boxes_map_class_names(tmp_path: Path) -> None:
    """FFA-Net boxes come from JSON objects, class names mapped via dataset_info."""
    doc = {"object_count": 2, "objects": [_obj(class_name="haze"), _obj(category_id=0)]}
    dataset = _dataset(tmp_path, "ffanet_indoor", _members(doc), classes=["smoke", "haze"])
    boxes = _boxes(dataset)
    assert boxes.shape == (MAX_FFANET_BOXES, 5)
    assert torch.allclose(boxes[:2], torch.tensor([[1.0, *BOX], [0.0, *BOX]]))
    assert not boxes[2:].any()


def test_ffanet_unknown_class_name_raises(tmp_path: Path) -> None:
    """An object naming no manifold class fails at decoding."""
    doc = {"object_count": 1, "objects": [_obj(class_name="fog")]}
    dataset = _dataset(tmp_path, "ffanet_indoor", _members(doc), classes=["smoke"])
    with pytest.raises(LabelDocumentError, match="'fog' is not a manifold class"):
        _ = dataset[0]


def test_ffanet_ignores_text_members(tmp_path: Path) -> None:
    """A .txt member gives no boxes: labels come only from JSON."""
    members = {**_members(), f"{KEY}.txt": b"0 0.5 0.5 0.2 0.2"}
    boxes = _boxes(_dataset(tmp_path, "ffanet_indoor", members))
    assert not boxes.any()


def test_face_without_json_raises(tmp_path: Path) -> None:
    """A face sample without a JSON label is reported, not an empty face."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.txt": b"0 0.5 0.4 0.2 0.3"}
    with pytest.raises(LabelDocumentError, match="has no JSON label"):
        _ = _dataset(tmp_path, "retinaface", members)[0]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"{broken", "unreadable JSON label"),
        (json_bytes({"landmark_source": "sfhq-110pt"}), "no objects list"),
        (json_bytes({"face_count": 1, "objects": [{"category_id": 0}]}), "4 numbers in 0..1"),
    ],
)
def test_face_malformed_json_raises(tmp_path: Path, raw: bytes, match: str) -> None:
    """Unreadable JSON, missing objects or an invalid box raise."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.json": raw}
    with pytest.raises(LabelDocumentError, match=match):
        _ = _dataset(tmp_path, "retinaface", members)[0]


def test_face_empty_objects_is_a_negative(tmp_path: Path) -> None:
    """An explicit empty objects list is a face-free image (confidence 0)."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.json": json_bytes({"face_count": 0, "objects": []})}
    target = _tensor_target(_dataset(tmp_path, "retinaface", members))
    assert target.shape == (FACE_TARGET_LEN,) and not target.any()


_CLASSIFIER = "universal_nsfw_classification"


def test_classification_name_from_dataset_info_classes(tmp_path: Path) -> None:
    """A model without registry class names maps ``class_name`` via dataset_info classes."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.json": json_bytes({"class_name": "b"})}
    dataset = _dataset(tmp_path, _CLASSIFIER, members, classes=["a", "b", "c"])
    dataset.spec.class_names = ()
    assert int(_tensor_target(dataset)) == 1


def test_classification_malformed_json_raises(tmp_path: Path) -> None:
    """A broken JSON label raises instead of becoming IGNORE_TARGET."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.json": b"[not json"}
    with pytest.raises(LabelDocumentError, match="unreadable JSON label"):
        _ = _dataset(tmp_path, _CLASSIFIER, members)[0]


def test_classification_text_member_is_unlabeled(tmp_path: Path) -> None:
    """A .txt member is not a label: the sample is unlabeled (IGNORE_TARGET)."""
    members = {f"{KEY}.webp": _IMAGE, f"{KEY}.txt": b"2"}
    assert int(_tensor_target(_dataset(tmp_path, _CLASSIFIER, members))) == IGNORE_TARGET
