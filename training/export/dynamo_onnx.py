"""ONNX graph capture with the ``torch.export``-based ``torch.onnx.export(dynamo=True)``.

Single responsibility: turn a module, sample inputs and an :class:`OnnxGraphSpec` into a
verified :class:`torch.onnx.ONNXProgram`. Weight layout and file names are decided elsewhere
(:mod:`training.export.onnx_weights`, :mod:`training.export.onnx_variants`).

``torch.export`` traces symbolically, so dynamic axes are real symbols rather than the
specialised trace of the legacy TorchScript exporter. A model may still branch in Python on
a shape (for example Swin's window shift, which switches off once a feature map is no larger
than the window); such a branch is resolved for the sample size and the graph is only valid
for inputs that take the same branch. :func:`build_verified_program` therefore runs the
exported graph in ONNX Runtime at a second, larger shape and compares it with PyTorch. When
the outputs disagree it pins the non-batch axes to the sample shape and exports again, so a
released graph never silently computes a different function at another resolution.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.onnx import ONNXProgram

from training.export.common import ExportError, OnnxGraphSpec
from training.utils.lazy_imports import load_onnxruntime

logger = logging.getLogger("lemtrain.export.dynamo_onnx")

AxisSpec = dict[int, Any]
"""Per-input ``torch.export`` dynamic shape: axis index -> axis name or ``Dim``."""


@dataclass(frozen=True)
class ProbeTolerance:
    """Tolerance of the shape-generalisation probe (FP32 graph against FP32 PyTorch)."""

    atol: float = 1e-3
    rtol: float = 1e-2


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of running an exported graph at the probe shape."""

    ok: bool
    max_abs_diff: float = float("nan")
    detail: str = ""

    def summary(self, tolerance: ProbeTolerance) -> str:
        """One line for the run log: verdict, worst difference and the tolerance used."""
        verdict = "passed" if self.ok else f"failed ({self.detail})"
        return (
            f"{verdict}, max abs diff {self.max_abs_diff:.3g} "
            f"(atol {tolerance.atol:g}, rtol {tolerance.rtol:g})"
        )


def report_parity(result: ProbeResult | None, tolerance: ProbeTolerance, shapes: str) -> str:
    """Print the ONNX FP32 parity line of an export and return it.

    The trainer configures no INFO handler, so a passed probe logged only through
    ``logging`` never reached the Kaggle log (run 6 of ``upn_v2`` showed no parity result).
    """
    if result is None:
        line = "[EXPORT] ONNX FP32 parity probe not run: the graph has no dynamic axes."
    else:
        line = f"[EXPORT] ONNX FP32 parity probe at {shapes}: {result.summary(tolerance)}"
    print(line, flush=True)
    return line


def _shapes_text(inputs: Sequence[torch.Tensor]) -> str:
    """Input shapes of a probe for the run log, e.g. ``1x3x192x192``."""
    return ", ".join("x".join(str(d) for d in t.shape) for t in inputs)


def flatten_outputs(outputs: Any) -> list[torch.Tensor]:
    """Tensors of a (nested) model output in the order the ONNX exporter flattens them."""
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, dict):
        return [t for value in outputs.values() for t in flatten_outputs(value)]
    if isinstance(outputs, (list, tuple)):
        return [t for value in outputs for t in flatten_outputs(value)]
    return []


def _takes_varargs(model: nn.Module) -> bool:
    """Whether ``model.forward`` collects its positional inputs in ``*args``."""
    params = inspect.signature(model.forward).parameters.values()
    return any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)


def dynamic_shapes_for(
    model: nn.Module, graph: OnnxGraphSpec, n_inputs: int
) -> tuple[Any, ...] | None:
    """``torch.export`` dynamic shapes for the inputs named in ``graph.dynamic_axes``.

    An axis name becomes the ONNX dimension name the first time it appears; later axes with
    the same name use ``Dim.DYNAMIC`` so ``torch.export`` infers the equality instead of
    being told the name twice. Output axes are inferred by the exporter.
    """
    if not graph.dynamic_axes:
        return None
    seen: set[str] = set()
    per_input: list[AxisSpec | None] = []
    for name in graph.input_names[:n_inputs]:
        axes = graph.dynamic_axes.get(name)
        if not axes:
            per_input.append(None)
            continue
        spec: AxisSpec = {}
        for axis, axis_name in axes.items():
            spec[axis] = torch.export.Dim.DYNAMIC if axis_name in seen else axis_name
            seen.add(axis_name)
        per_input.append(spec)
    per_input.extend([None] * (n_inputs - len(per_input)))
    if all(spec is None for spec in per_input):
        return None
    return (tuple(per_input),) if _takes_varargs(model) else tuple(per_input)


def capture_program(
    model: nn.Module, inputs: tuple[torch.Tensor, ...], graph: OnnxGraphSpec
) -> ONNXProgram:
    """Export ``model`` with ``torch.onnx.export(dynamo=True)`` (no file is written)."""
    with torch.no_grad():
        program = torch.onnx.export(
            model,
            inputs,
            dynamo=True,
            input_names=graph.input_names,
            output_names=graph.output_names,
            opset_version=graph.opset,
            dynamic_shapes=dynamic_shapes_for(model, graph, len(inputs)),
            optimize=True,
            # The exporter's progress lines contain non-ASCII symbols that a cp1252 console
            # cannot encode; progress is reported through this module's logger instead.
            verbose=False,
        )
    if program is None:
        raise ExportError("ONNX_EXPORT_FAILED", "torch.onnx.export returned no ONNX program")
    return program


def _grown_size(size: int) -> int:
    """A different, larger probe size for an axis of ``size``."""
    return size + max(1, size // 2)


def probe_inputs(inputs: Sequence[torch.Tensor], graph: OnnxGraphSpec) -> tuple[torch.Tensor, ...]:
    """Inputs with every dynamic axis grown; axes sharing a name grow to the same size.

    Floating inputs are random; integer inputs (indices such as a pair id) repeat the sample
    values so they stay inside the model's valid range.
    """
    axes_by_input = graph.dynamic_axes or {}
    sizes: dict[str, int] = {}
    probes: list[torch.Tensor] = []
    for index, tensor in enumerate(inputs):
        name = graph.input_names[index] if index < len(graph.input_names) else ""
        shape = list(tensor.shape)
        for axis, axis_name in axes_by_input.get(name, {}).items():
            shape[axis] = sizes.setdefault(axis_name, _grown_size(shape[axis]))
        if tensor.is_floating_point():
            probes.append(torch.randn(shape, dtype=tensor.dtype, device=tensor.device))
        else:
            flat = tensor.flatten()
            idx = torch.arange(int(torch.Size(shape).numel()), device=tensor.device) % flat.numel()
            probes.append(flat[idx].reshape(shape))
    return tuple(probes)


PROBE_DEVICE = torch.device("cpu")
"""Device the probe compares outputs on: where ONNX Runtime's CPU provider returns them."""


def _cpu_session(model: str | bytes) -> Any:
    """ONNX Runtime session on the CPU provider (deterministic, available everywhere)."""
    ort = load_onnxruntime()
    return ort.InferenceSession(model, providers=["CPUExecutionProvider"])


def ort_run_errors() -> tuple[type[Exception], ...]:
    """Exception types ONNX Runtime raises when a graph cannot run on given inputs."""
    state = load_onnxruntime().capi.onnxruntime_pybind11_state
    return (
        state.Fail,
        state.InvalidArgument,
        state.RuntimeException,
        state.InvalidGraph,
        state.NotImplemented,
    )


def run_probe(
    model: nn.Module,
    program: ONNXProgram,
    inputs: tuple[torch.Tensor, ...],
    tolerance: ProbeTolerance,
) -> ProbeResult:
    """Compare ``program`` (ONNX Runtime, CPU) with ``model`` (PyTorch) on ``inputs``."""
    with torch.no_grad():
        expected = flatten_outputs(model(*inputs))
    program.initialize_inference_session(_cpu_session)
    try:
        actual = list(program(*inputs))
    except ort_run_errors() as exc:
        return ProbeResult(ok=False, detail=f"ONNX Runtime rejected the probe shape: {exc}")
    finally:
        program.release()
    return compare_outputs(expected, actual, tolerance)


def compare_outputs(
    expected: Sequence[torch.Tensor], actual: Sequence[torch.Tensor], tolerance: ProbeTolerance
) -> ProbeResult:
    """Compare PyTorch outputs with ONNX Runtime outputs on the CPU.

    ONNX Runtime returns CPU tensors while a model trained on a GPU returns CUDA tensors;
    both sides are moved to the CPU before they meet (an end-of-epoch export on Kaggle
    failed with "Expected all tensors to be on the same device").
    """
    if len(actual) != len(expected):
        return ProbeResult(ok=False, detail=f"{len(actual)} outputs, expected {len(expected)}")
    worst = 0.0
    for want_any, got in zip(expected, actual, strict=True):
        want = want_any.detach().to(PROBE_DEVICE)
        got_t = got.to(device=PROBE_DEVICE, dtype=want.dtype)
        if want.shape != got_t.shape:
            return ProbeResult(
                ok=False, detail=f"shape {tuple(got_t.shape)} != {tuple(want.shape)}"
            )
        if want.numel():
            worst = max(worst, float((want - got_t).abs().max()))
        if not torch.allclose(got_t, want, atol=tolerance.atol, rtol=tolerance.rtol):
            return ProbeResult(ok=False, max_abs_diff=worst, detail="outputs differ")
    return ProbeResult(ok=True, max_abs_diff=worst)


def batch_only(graph: OnnxGraphSpec) -> OnnxGraphSpec:
    """``graph`` with every dynamic axis except axis 0 (batch) pinned to the sample shape."""
    axes = {
        name: {axis: label for axis, label in spec.items() if axis == 0}
        for name, spec in (graph.dynamic_axes or {}).items()
    }
    return replace(graph, dynamic_axes={k: v for k, v in axes.items() if v} or None)


@dataclass
class VerifiedProgram:
    """An exported FP32 program and the graph spec that it was verified with."""

    program: ONNXProgram
    graph: OnnxGraphSpec
    probes: list[ProbeResult] = field(default_factory=list)


def build_verified_program(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    graph: OnnxGraphSpec,
    tolerance: ProbeTolerance | None = None,
    fallbacks: Sequence[Callable[[OnnxGraphSpec], OnnxGraphSpec]] = (batch_only,),
) -> VerifiedProgram:
    """Export ``model`` and verify that its dynamic axes generalise beyond the sample shape.

    Each failed probe narrows the spec with the next entry of ``fallbacks``; the export
    fails with :class:`ExportError` when no spec produces a graph that matches PyTorch.
    """
    tolerance = tolerance or ProbeTolerance()
    specs = [graph, *(narrow(graph) for narrow in fallbacks)]
    probes: list[ProbeResult] = []
    for spec in specs:
        program = capture_program(model, inputs, spec)
        if not spec.dynamic_axes:
            report_parity(None, tolerance, "")
            return VerifiedProgram(program, spec, probes)
        shaped = probe_inputs(inputs, spec)
        result = run_probe(model, program, shaped, tolerance)
        probes.append(result)
        if result.ok:
            report_parity(result, tolerance, _shapes_text(shaped))
            return VerifiedProgram(program, spec, probes)
        logger.warning(
            "Exported graph does not generalise over dynamic axes %s (%s, max abs diff %.3g); "
            "pinning more axes to the sample shape.",
            spec.dynamic_axes,
            result.detail,
            result.max_abs_diff,
        )
    raise ExportError(
        "ONNX_DYNAMIC_AXES_MISMATCH",
        f"No dynamic-axes spec produced a graph matching PyTorch: {[p.detail for p in probes]}",
    )
