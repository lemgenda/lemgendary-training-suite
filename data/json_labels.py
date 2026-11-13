"""Read the JSON label document of a shard sample (the one label format in the shards).

The dataset compiler writes every label as ``<key>.json`` (lemgendary-datasets
``utils/label_json.py`` defines the schema). The keys read here:

* ``objects``: ``[{"category_id": 3, "bbox_cxcywh_normalized": [cx, cy, w, h]}, ...]``
  (detection, the auto-labelled restoration tasks and RetinaFace faces). An object may
  name its class with ``class_name`` instead of ``category_id``; the name maps to an id
  through the manifold's ``dataset_info.yaml`` ``classes`` list.
* ``class_id`` / ``class_name``: a classification label.
* ``caption``: a diffusion caption.

A document that breaks the schema raises :class:`LabelDocumentError` naming what is
wrong, so a damaged label never trains as a default target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from training.data.containers.base import JSON_ERROR_KEY, JSON_METADATA_KEY

BOX_KEY = "bbox_cxcywh_normalized"
"""Object key of a normalised ``cx cy w h`` box."""

BOX_VALUES = 4

_CLASS_LABEL_KEYS = ("class_id", "class_name", "label", "class")
"""Classification label keys: the schema's two, then those of older JSON sidecars."""


class LabelDocumentError(ValueError):
    """A sample's JSON label is unreadable or breaks the label schema."""


YoloRow = tuple[int, float, float, float, float]
"""One object as a YOLO row: class id and the normalised ``cx cy w h`` box."""


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def object_class_id(obj: Mapping[str, Any], classes: Sequence[str]) -> int:
    """Class id of one object: ``category_id``, else ``class_name`` looked up in ``classes``.

    Raises :class:`LabelDocumentError` for a negative or non-integer id, an unknown name or
    an object with neither key.
    """
    category = obj.get("category_id")
    if category is not None:
        if isinstance(category, bool) or not isinstance(category, int) or category < 0:
            raise LabelDocumentError(f"category_id {category!r} is not a non-negative integer")
        return category
    name = obj.get("class_name")
    if not isinstance(name, str) or not name.strip():
        raise LabelDocumentError("object has neither category_id nor class_name")
    if name.strip() not in classes:
        known = ", ".join(classes) if classes else "none (dataset_info.yaml has no classes)"
        raise LabelDocumentError(f"class_name {name!r} is not a manifold class (known: {known})")
    return list(classes).index(name.strip())


def object_box(obj: Mapping[str, Any]) -> list[float]:
    """The object's ``bbox_cxcywh_normalized`` box; :class:`LabelDocumentError` when invalid."""
    box = obj.get(BOX_KEY)
    if not (
        isinstance(box, list)
        and len(box) == BOX_VALUES
        and all(_is_number(v) and 0.0 <= v <= 1.0 for v in box)
    ):
        raise LabelDocumentError(f"{BOX_KEY} {box!r} is not 4 numbers in 0..1")
    return [float(v) for v in box]


def label_objects(doc: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """The document's ``objects`` (None when it has no ``objects`` key).

    Raises :class:`LabelDocumentError` when ``objects`` is not a list of objects.
    """
    objects = doc.get("objects")
    if objects is None:
        return None
    if not isinstance(objects, list) or not all(isinstance(o, Mapping) for o in objects):
        raise LabelDocumentError("objects is not a list of objects")
    return objects


def object_boxes(doc: Mapping[str, Any]) -> list[list[float]] | None:
    """The box of every object of ``doc`` (None when it has no ``objects`` key).

    Classes are not read (a single-class face label). Raises :class:`LabelDocumentError`
    for an object without a valid box.
    """
    objects = label_objects(doc)
    if objects is None:
        return None
    boxes = []
    for index, obj in enumerate(objects):
        try:
            boxes.append(object_box(obj))
        except LabelDocumentError as exc:
            raise LabelDocumentError(f"objects[{index}]: {exc}") from exc
    return boxes


def yolo_rows(doc: Mapping[str, Any], classes: Sequence[str] = ()) -> list[YoloRow] | None:
    """Every object of ``doc`` as a YOLO row (None when it has no ``objects`` key).

    An empty ``objects`` list is an image without objects (``[]``). Raises
    :class:`LabelDocumentError` for an object without a valid box or class.
    """
    objects = label_objects(doc)
    if objects is None:
        return None
    rows: list[YoloRow] = []
    for index, obj in enumerate(objects):
        try:
            cls = object_class_id(obj, classes)
            cx, cy, w, h = object_box(obj)
            rows.append((cls, cx, cy, w, h))
        except LabelDocumentError as exc:
            raise LabelDocumentError(f"objects[{index}]: {exc}") from exc
    return rows


def class_label(doc: Any) -> Any:
    """The classification label of ``doc`` (an id or a class name); None when it has none."""
    if not isinstance(doc, Mapping):
        return None
    for key in _CLASS_LABEL_KEYS:
        value = doc.get(key)
        if value is not None:
            return value
    return None


def caption(doc: Any) -> str | None:
    """The document's ``caption``; None when it has none.

    Raises :class:`LabelDocumentError` when ``caption`` is not a string.
    """
    if not isinstance(doc, Mapping) or doc.get("caption") is None:
        return None
    text = doc["caption"]
    if not isinstance(text, str):
        raise LabelDocumentError(f"caption {text!r} is not a string")
    return text.strip() or None


def sample_document(metadata: Mapping[str, Any], where: str) -> Mapping[str, Any] | None:
    """The sample's JSON label document; None when the sample has no ``.json`` member.

    Raises :class:`LabelDocumentError` naming ``where`` when the member was unreadable
    (:data:`JSON_ERROR_KEY`) or is not a JSON object.
    """
    error = metadata.get(JSON_ERROR_KEY)
    if error is not None:
        raise LabelDocumentError(f"{where}: unreadable JSON label ({error})")
    doc = metadata.get(JSON_METADATA_KEY)
    if doc is None:
        return None
    if not isinstance(doc, Mapping):
        raise LabelDocumentError(f"{where}: the JSON label is not an object")
    return doc
