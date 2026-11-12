"""ONNX weight layout policy and writers (single source of truth for the sidecar rules).

Release rule for the ONNX variants:

* FP32 ONNX (``<Stem>_FP32.onnx``) always stores its weights in an external
  ``<Stem>_FP32.onnx.data`` sidecar (:attr:`WeightLayout.SIDECAR`).
* FP16 ONNX (``<Stem>.onnx``) embeds its weights in the ``.onnx`` file unless the serialized
  model would exceed the protobuf limit of 2 GB, in which case the weights go to a
  ``<Stem>.onnx.data`` sidecar exactly like FP32 (:attr:`WeightLayout.EMBEDDED`).

Every ONNX file the export subsystem writes goes through :func:`save_program` (graphs from
``torch.onnx.export(dynamo=True)``) or :func:`save_proto` (graphs from converters and
third-party exporters), so the rule is decided in one place.
"""

from __future__ import annotations

import logging
import math
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
from torch.onnx import ONNXProgram

from training.cloud.variations import onnx_sidecar_name

logger = logging.getLogger("lemtrain.export.onnx_weights")

PROTOBUF_LIMIT_BYTES: int = onnx.checker.MAXIMUM_PROTOBUF
"""Largest ONNX file protobuf can serialize (2 GB); larger models need a weights sidecar.

Read at call time, so tests can lower it to exercise the oversize branch without a 2 GB model.
"""


class WeightLayout(Enum):
    """Where an ONNX variant keeps its weights."""

    SIDECAR = "sidecar"
    """Always in the external ``<file>.data`` sidecar (FP32)."""
    EMBEDDED = "embedded"
    """In the ``.onnx`` file, unless the model exceeds :data:`PROTOBUF_LIMIT_BYTES` (FP16)."""


def sidecar_path(onnx_path: Path) -> Path:
    """Path of the external weights sidecar that belongs to ``onnx_path``."""
    return onnx_path.with_name(onnx_sidecar_name(onnx_path.name))


def uses_sidecar(layout: WeightLayout, model_bytes: int) -> bool:
    """Whether a model of ``model_bytes`` serialized bytes is written with a sidecar."""
    return layout is WeightLayout.SIDECAR or model_bytes > PROTOBUF_LIMIT_BYTES


def proto_weight_bytes(model: onnx.ModelProto) -> int:
    """Total size of the model's initializers in bytes (embedded or external)."""
    total = 0
    for tensor in model.graph.initializer:
        itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        total += math.prod(tensor.dims) * itemsize
    return total


def _clear_sidecar(onnx_path: Path) -> Path:
    """Delete a stale sidecar of ``onnx_path`` (onnx appends to existing data files)."""
    data_path = sidecar_path(onnx_path)
    if data_path.exists():
        data_path.unlink()
        logger.debug("Removed stale ONNX weights sidecar %s", data_path)
    return data_path


def _saved(onnx_path: Path, external: bool, layout: WeightLayout) -> Path | None:
    """Log the written layout and return the sidecar path when one was written."""
    data_path = sidecar_path(onnx_path)
    if external:
        if not data_path.is_file():
            logger.warning("ONNX model %s has no weights to store in a sidecar", onnx_path.name)
            return None
        logger.info(
            "Saved %s with external weights sidecar %s (%s layout)",
            onnx_path.name,
            data_path.name,
            layout.value,
        )
        return data_path
    logger.info("Saved %s with embedded weights", onnx_path.name)
    return None


def save_proto(model: onnx.ModelProto, onnx_path: Path, layout: WeightLayout) -> Path | None:
    """Write ``model`` to ``onnx_path`` with ``layout``; return the sidecar path if one is written.

    ``model`` must hold its tensor data in memory (``onnx.load`` loads external data by default).
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = _clear_sidecar(onnx_path)
    external = uses_sidecar(layout, max(model.ByteSize(), proto_weight_bytes(model)))
    if external:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
    else:
        onnx.save_model(model, str(onnx_path))
    return _saved(onnx_path, external, layout)


def save_program(program: ONNXProgram, onnx_path: Path, layout: WeightLayout) -> Path | None:
    """Write a ``torch.onnx`` program with ``layout``; return the sidecar path if one is written.

    The embedded branch serializes the proto directly: ``ONNXProgram.save`` would move weights
    out at its own 1.5 GB threshold instead of the 2 GB protobuf limit this policy uses.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_sidecar(onnx_path)
    if layout is WeightLayout.SIDECAR:
        program.save(str(onnx_path), external_data=True)
        return _saved(onnx_path, True, layout)
    model = program.model_proto
    return save_proto(model, onnx_path, layout)


def relocate_onnx(src: Path, dest: Path) -> Path | None:
    """Move an ONNX file (and its sidecar) to ``dest``; return the sidecar's new path, if any.

    The sidecar location is recorded inside the ``.onnx`` file, so a renamed model with a
    sidecar is re-saved under its new name instead of being moved byte for byte.
    """
    src_data = sidecar_path(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return src_data if src_data.is_file() else None
    if not src_data.is_file():
        _clear_sidecar(dest)
        shutil.move(str(src), str(dest))
        return None
    if src.name == dest.name:
        dest_data = _clear_sidecar(dest)
        shutil.move(str(src), str(dest))
        shutil.move(str(src_data), str(dest_data))
        return dest_data
    # ``load_external_data`` reads the sidecar into memory and clears the location markers.
    model = onnx.load(str(src), load_external_data=True)
    logger.info("Re-saving %s as %s to rewrite its sidecar location", src.name, dest.name)
    new_data = save_proto(model, dest, WeightLayout.SIDECAR)
    src.unlink()
    src_data.unlink()
    return new_data
