"""The two ONNX release variants (FP32 with sidecar, FP16 embedded) and their writer.

Together with ``<Stem>.pt`` (:func:`training.export.torch_standalone.export_torch_standalone`)
they form the three release exports of every model:

1. ``<Stem>.pt``: PyTorch FP32, self-contained.
2. ``<Stem>_FP32.onnx`` + ``<Stem>_FP32.onnx.data``: ONNX FP32, weights always external.
3. ``<Stem>.onnx``: ONNX FP16, weights embedded unless they exceed 2 GB, then in
   ``<Stem>.onnx.data`` like FP32.

File suffixes come from :mod:`training.cloud.variations` and the sidecar rule from
:mod:`training.export.onnx_weights`; graphs are captured by :mod:`training.export.dynamo_onnx`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from training.cloud.variations import ONNX_FP16_SUFFIX, ONNX_FP32_SUFFIX
from training.export.base import EXPORT_ERRORS
from training.export.common import OnnxGraphSpec
from training.export.dynamo_onnx import (
    VerifiedProgram,
    build_verified_program,
    capture_program,
    flatten_outputs,
)
from training.export.onnx_weights import WeightLayout, save_program
from training.export.precision import reduced_precision_copy
from training.utils.lazy_imports import load_onnx_float16

logger = logging.getLogger("lemtrain.export.onnx_variants")


@dataclass(frozen=True)
class OnnxVariant:
    """One ONNX release variant: result key, precision, weight layout and file suffix."""

    key: str
    dtype: torch.dtype
    layout: WeightLayout
    suffix: str

    @property
    def data_key(self) -> str:
        """Result key of the variant's weights sidecar."""
        return f"{self.key}_data"

    def file_name(self, stem: str) -> str:
        """Release file name of this variant for the model ``stem``."""
        return f"{stem}{self.suffix}"


ONNX_FP32_VARIANT = OnnxVariant("onnx_fp32", torch.float32, WeightLayout.SIDECAR, ONNX_FP32_SUFFIX)
ONNX_FP16_VARIANT = OnnxVariant("onnx_fp16", torch.float16, WeightLayout.EMBEDDED, ONNX_FP16_SUFFIX)
ONNX_VARIANTS: tuple[OnnxVariant, ...] = (ONNX_FP32_VARIANT, ONNX_FP16_VARIANT)


def cast_inputs(inputs: tuple[torch.Tensor, ...], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Cast floating inputs to ``dtype``; integer inputs (indices) are left unchanged."""
    return tuple(t.to(dtype) if t.is_floating_point() else t for t in inputs)


def convert_proto_to_fp16(
    model: onnx.ModelProto, *, keep_io_types: bool, disable_shape_infer: bool = False
) -> onnx.ModelProto:
    """FP16 copy of an FP32 ONNX graph, for graphs not captured from a PyTorch module.

    ``disable_shape_infer`` skips ONNX shape inference, which cannot run on models above
    the 2 GB protobuf limit.
    """
    float16: Any = load_onnx_float16()
    return float16.convert_float_to_float16(
        model, keep_io_types=keep_io_types, disable_shape_infer=disable_shape_infer
    )


class Float32Io(nn.Module):
    """Runs a reduced-precision module behind FP32 inputs and outputs.

    The exported graph casts at its boundary (like ``keep_io_types`` of the ONNX float16
    converter), so callers that can only feed FP32 tensors (e.g. MetaTrader 5) can still use
    the FP16 variant.
    """

    def __init__(self, inner: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner = inner
        self.dtype = dtype

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Cast floating inputs to the inner precision and every output back to FP32."""
        outputs = self.inner(*cast_inputs(args, self.dtype))
        return tuple(t.float() for t in flatten_outputs(outputs))


@dataclass
class OnnxVariantWriter:
    """Writes the ONNX variants of one model from a single verified graph spec.

    The FP32 graph is captured and verified once (:func:`build_verified_program`); the FP16
    graph is captured from a half-precision copy of the model with the same spec, so both
    variants share their input names and dynamic axes. The caller's model is never cast.
    ``fp32_io`` keeps FP32 inputs and outputs on reduced-precision variants (see
    :class:`Float32Io`). ``finalizers`` maps a variant key to a step run on the written file
    (e.g. the MT5 IR stamp of the Forex FP16 artefact), so a model family adapts its release
    files without changing the writer.
    """

    model: nn.Module
    inputs: tuple[torch.Tensor, ...]
    graph: OnnxGraphSpec
    fp32_io: bool = False
    finalizers: Mapping[str, Callable[[Path], None]] = field(default_factory=dict)
    _verified: VerifiedProgram | None = field(default=None, init=False, repr=False)

    def verified(self) -> VerifiedProgram:
        """The FP32 program and its verified graph spec (exported on first use)."""
        if self._verified is None:
            fp32_inputs = cast_inputs(self.inputs, torch.float32)
            self._verified = build_verified_program(self.model, fp32_inputs, self.graph)
        return self._verified

    def _program(self, variant: OnnxVariant) -> Any:
        """The captured program of ``variant`` (FP32 reuses the verified one)."""
        verified = self.verified()
        if variant.dtype == torch.float32:
            return verified.program
        model_copy = reduced_precision_copy(self.model, variant.dtype)
        if self.fp32_io:
            wrapped = Float32Io(model_copy, variant.dtype).eval()
            return capture_program(wrapped, cast_inputs(self.inputs, torch.float32), verified.graph)
        return capture_program(model_copy, cast_inputs(self.inputs, variant.dtype), verified.graph)

    def _verify_canary(self, program: Any, variant: OnnxVariant) -> None:
        """Run a lightweight NaN/Inf canary on FP16 export instead of expensive CPU parity probe."""
        if variant.dtype != torch.float16:
            return
        test_inputs = cast_inputs(self.inputs, torch.float32 if self.fp32_io else torch.float16)
        try:
            outputs = list(program(*test_inputs))
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    if not torch.isfinite(out).all():
                        logger.warning(
                            "[CANARY] Non-finite output (NaN/Inf) detected in %s at output %d",
                            variant.key,
                            i,
                        )
                    else:
                        logger.debug("[CANARY] %s output %d verified finite.", variant.key, i)
        except Exception as exc:
            logger.debug("[CANARY] Skipping runtime canary for %s: %s", variant.key, exc)

    def write(self, variant: OnnxVariant, path: Path) -> dict[str, Path]:
        """Export ``variant`` to ``path``; returns ``{key: path}`` plus ``{data_key: sidecar}``."""
        logger.info("Exporting %s ONNX graph to %s", variant.key, path)
        prog = self._program(variant)
        self._verify_canary(prog, variant)
        sidecar = save_program(prog, path, variant.layout)
        finalize = self.finalizers.get(variant.key)
        if finalize is not None:
            finalize(path)
        files = {variant.key: path}
        if sidecar is not None:
            files[variant.data_key] = sidecar
        return files

    def write_all(
        self, directory: Path, stem: str, variants: tuple[OnnxVariant, ...] = ONNX_VARIANTS
    ) -> dict[str, Path]:
        """Export ``variants`` into ``directory`` under their release names for ``stem``."""
        files: dict[str, Path] = {}
        for variant in variants:
            files.update(self.write(variant, directory / variant.file_name(stem)))
        return files


def write_release_variants(
    writer: OnnxVariantWriter, directory: Path, stem: str
) -> dict[str, Path]:
    """Write every ONNX release variant; a failed variant is logged and the others still run."""
    files: dict[str, Path] = {}
    for variant in ONNX_VARIANTS:
        try:
            files.update(writer.write(variant, directory / variant.file_name(stem)))
        except EXPORT_ERRORS as exc:
            logger.error("Failed exporting %s for %s: %s", variant.key, stem, exc)
    return files
