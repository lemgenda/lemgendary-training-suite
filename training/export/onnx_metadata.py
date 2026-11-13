"""Model metadata written into exported image-model ONNX files (``metadata_props``).

An ONNX file then describes itself without the registry: which model and task it is, the
``[C, H, W]`` input it was exported with, the image preprocessing it was trained with
(:mod:`training.registry.preprocessing`) and, for a classifier, the class name of each
output logit (:mod:`training.registry.class_names`). Every value is a string; the lists are
JSON. :func:`metadata_finalizer` stamps a written file (the ``OnnxVariantWriter``
finalizer hook), keeping an external-data sidecar untouched.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from training.registry.class_names import class_names
from training.registry.preprocessing import image_preprocessing, task_type_of
from training.utils.lazy_imports import load_onnx

CLASS_NAMES_PROP = "class_names"
INPUT_SIZE_PROP = "input_size"


def onnx_metadata(
    model_key: str, model_info: Mapping[str, Any] | None, input_shape: Sequence[int]
) -> dict[str, str]:
    """``metadata_props`` of ``model_key``: identity, task, input size, preprocessing, classes.

    ``input_shape`` is the export sample's ``[C, H, W]`` (the batch axis excluded).
    """
    info = model_info or {}
    props = {
        "model_key": model_key,
        "task": task_type_of(info),
        INPUT_SIZE_PROP: json.dumps([int(dim) for dim in input_shape]),
        **image_preprocessing(info).metadata(),
    }
    names = class_names(info)
    if names:
        props[CLASS_NAMES_PROP] = json.dumps(list(names))
    return props


def stamp_metadata(path: Path, props: Mapping[str, str]) -> None:
    """Set ``props`` in the ``metadata_props`` of the ONNX file at ``path`` (others kept).

    The graph is rewritten without loading external weights, so a ``.onnx.data`` sidecar
    stays as it is and keeps being referenced.
    """
    onnx = load_onnx()
    model = onnx.load(str(path), load_external_data=False)
    kept = [prop for prop in model.metadata_props if prop.key not in props]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))


def read_metadata(path: Path) -> dict[str, str]:
    """The ``metadata_props`` of the ONNX file at ``path``."""
    model = load_onnx().load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def metadata_finalizer(props: Mapping[str, str]) -> Callable[[Path], None]:
    """Writer finalizer that stamps ``props`` into a written ONNX file."""
    frozen = dict(props)

    def _stamp(path: Path) -> None:
        stamp_metadata(path, frozen)

    return _stamp
