"""MetaTrader 5 ONNX compatibility of the Forex EA artefact (R4-C1).

MetaTrader 5 (build 4230 and later) runs ONNX models on its bundled ONNX Runtime 1.17, which
reads ai.onnx opsets up to :data:`MT5_MAX_OPSET` (20) and IR versions up to
:data:`MT5_MAX_IR_VERSION` (9). ``torch.onnx.export(dynamo=True)`` with onnx 1.23 stamps IR 10,
so ORT 1.17 refuses the file ("Unsupported model IR version: 10, max supported IR version: 9").
``OnnxCreate`` / ``OnnxCreateFromBuffer`` load one file or buffer, so the artefact must also
embed its weights (no ``.onnx.data`` sidecar).

The suite keeps opset 18 (:data:`training.export.common.DEFAULT_ONNX_OPSET`) and only lowers
the IR stamp of the MT5 artefact. That is lossless when the graph uses none of the additions
of IR 10 and later, which :func:`ir9_blockers` checks: no local functions or function
overloads, no 4-bit, 2-bit or 8-bit float tensor types, no per-node or per-graph metadata
(IR 10 fields; the exporter's stack-trace metadata is dropped). The lowered model must then
pass ``onnx.checker.check_model(full_check=True)``.

The runtime version per MT5 build is not published, so the live check stays manual: run the
EA in the Strategy Tester (build 5572 and later log the runtime with ``ONNX_LOGLEVEL_INFO``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import onnx
from onnx import TensorProto

from training.export.common import ExportError
from training.export.onnx_weights import WeightLayout, save_proto, sidecar_path

logger = logging.getLogger("lemtrain.export.mt5_compat")

MT5_MAX_IR_VERSION = 9
"""Largest ONNX IR version ONNX Runtime 1.17 (bundled with MT5 build 4230) reads."""
MT5_MAX_OPSET = 20
"""Largest ai.onnx opset ONNX Runtime 1.17 implements."""
MT5_ORT_VERSION = "1.17.3"
"""ONNX Runtime release matching MT5's bundled runtime (used by the optional load test)."""

_DEFAULT_DOMAINS = ("", "ai.onnx")
_POST_IR9_TYPES: frozenset[int] = frozenset(
    getattr(TensorProto, name)
    for name in (
        "INT4",
        "UINT4",
        "INT2",
        "UINT2",
        "FLOAT4E2M1",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
        "FLOAT8E8M0",
    )
    if hasattr(TensorProto, name)
)
"""Element types an MT5 graph must not use: IR 10+ additions, and 8-bit floats ORT 1.17 lacks."""


def _value_types(graph: onnx.GraphProto) -> Iterator[tuple[str, int]]:
    """``(name, elem_type)`` of every typed tensor in ``graph`` (I/O, value info, weights)."""
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.type.HasField("tensor_type"):
            yield value.name, value.type.tensor_type.elem_type
    for tensor in graph.initializer:
        yield tensor.name, tensor.data_type


def _node_blockers(graph: onnx.GraphProto) -> Iterator[str]:
    """IR 10 features used by the nodes of ``graph``."""
    for node in graph.node:
        label = node.name or node.op_type
        if node.overload:
            yield f"node {label} calls function overload {node.overload!r}"
        if node.domain not in _DEFAULT_DOMAINS:
            yield f"node {label} uses custom domain {node.domain!r}"
        for attr in node.attribute:
            is_cast_target = node.op_type in ("Cast", "CastLike") and attr.name == "to"
            if is_cast_target and attr.i in _POST_IR9_TYPES:
                yield f"node {label} casts to unsupported type {attr.i}"
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type in _POST_IR9_TYPES:
                yield f"node {label} attribute {attr.name} has unsupported type"


def ir9_blockers(model: onnx.ModelProto) -> list[str]:
    """Reasons ``model`` cannot be stamped IR 9 losslessly (empty when it can)."""
    blockers: list[str] = []
    if model.functions:
        blockers.append(f"{len(model.functions)} local function(s)")
    for opset in model.opset_import:
        if opset.domain in _DEFAULT_DOMAINS and opset.version > MT5_MAX_OPSET:
            blockers.append(f"ai.onnx opset {opset.version} > {MT5_MAX_OPSET}")
    for name, elem_type in _value_types(model.graph):
        if elem_type in _POST_IR9_TYPES:
            blockers.append(f"tensor {name} has unsupported type {elem_type}")
    blockers.extend(_node_blockers(model.graph))
    return blockers


def _drop_ir10_metadata(model: onnx.ModelProto) -> int:
    """Clear the IR 10 per-node and per-graph metadata; returns the number of nodes cleared."""
    cleared = 0
    for node in model.graph.node:
        if len(node.metadata_props):
            del node.metadata_props[:]
            cleared += 1
    del model.graph.metadata_props[:]
    return cleared


def stamp_mt5_ir(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` lowered to IR :data:`MT5_MAX_IR_VERSION` in place, checked; refuses IR 10 use."""
    blockers = ir9_blockers(model)
    if blockers:
        raise ExportError(
            "MT5_IR_INCOMPATIBLE",
            "The graph cannot run on MT5's ONNX Runtime 1.17: " + "; ".join(blockers),
        )
    cleared = _drop_ir10_metadata(model)
    previous = model.ir_version
    model.ir_version = min(model.ir_version, MT5_MAX_IR_VERSION)
    try:
        onnx.checker.check_model(model, full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as exc:
        raise ExportError("MT5_ONNX_CHECK_FAILED", f"IR {model.ir_version} graph: {exc}") from exc
    logger.info(
        "MT5 artefact stamped IR %d (was %d), %d node metadata entries dropped",
        model.ir_version,
        previous,
        cleared,
    )
    return model


def make_mt5_compatible(onnx_path: Path) -> None:
    """Rewrite ``onnx_path`` as the MT5 artefact: one file, embedded weights, IR 9, checked."""
    if sidecar_path(onnx_path).exists():
        raise ExportError(
            "MT5_SIDECAR",
            f"{onnx_path.name} keeps its weights in a sidecar; MT5 OnnxCreate needs one file.",
        )
    model = stamp_mt5_ir(onnx.load(str(onnx_path)))
    if save_proto(model, onnx_path, WeightLayout.EMBEDDED) is not None:
        raise ExportError(
            "MT5_SIDECAR",
            f"{onnx_path.name} exceeds the 2 GB single-file limit; MT5 cannot load it.",
        )
