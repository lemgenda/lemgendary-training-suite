"""Common export helpers, path resolution, and model wrapping."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from models.nima import SoftmaxWrapper
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.common")


class ExportError(Exception):
    """Raised when model export fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def resolve_export_paths(
    model_key: str,
    model_info: dict[str, Any],
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> tuple[str, Path]:
    """Compute base artifact name and production destination directory."""
    root = project_root or get_project_root()
    cfg = config or {}

    model_filename = model_info.get("filename", model_key)
    base_name = f"LemGendary{model_filename}"

    export_dir_rel = cfg.get("export_dir", "../LemGendaryModels")
    production_dir = (root / export_dir_rel / model_key).resolve()
    production_dir.mkdir(parents=True, exist_ok=True)

    return base_name, production_dir


def resolve_checkpoint_file(
    model_key: str,
    production_dir: Path,
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
    user_checkpoint: str | Path | None = None,
) -> Path | None:
    """Locate the best, latest, or user-specified .pth checkpoint file."""
    if user_checkpoint:
        cand = Path(user_checkpoint).resolve()
        if cand.exists() and cand.is_file():
            return cand

    hub_ckpt_dir = production_dir / "checkpoints"
    for suffix in ("_best.pth", "_latest.pth", "_progress.pth"):
        cand = hub_ckpt_dir / f"{model_key}{suffix}"
        if cand.exists() and cand.is_file():
            return cand

    root = project_root or get_project_root()
    cfg = config or {}
    legacy_dir = (root / cfg.get("checkpoint_dir", "checkpoints")).resolve()
    legacy_candidate = legacy_dir / f"{model_key}_best.pth"
    if legacy_candidate.exists() and legacy_candidate.is_file():
        return legacy_candidate

    return None


def extract_input_dimensions(
    model_info: dict[str, Any],
    default_size: int = 256,
) -> tuple[int, int]:
    """Extract height and width from model_info configuration."""
    size_raw = model_info.get("input_size", default_size)
    if size_raw is None:
        size_raw = default_size

    if isinstance(size_raw, (list, tuple)):
        if len(size_raw) == 3:
            h, w = int(size_raw[1]), int(size_raw[2])
        elif len(size_raw) >= 2:
            h, w = int(size_raw[0]), int(size_raw[1])
        else:
            h = w = int(size_raw[0])
    else:
        h = w = int(size_raw)

    return min(h, 512), min(w, 512)


def wrap_quality_model(model: nn.Module, model_info: dict[str, Any]) -> nn.Module:
    """Applies softmax temperature wrapper for quality scoring models if needed."""
    if model_info.get("dataset_type") == "quality":
        try:
            temp_raw = getattr(model, "softmax_temp", torch.tensor(1.0))
            temp = float(temp_raw.item() if isinstance(temp_raw, torch.Tensor) else temp_raw)
            if temp == 1.0:
                stab = model_info.get("stabilizers", {})
                temp = float(stab.get("softmax_temp", 1.0))
            logger.info("Applying production SoftmaxWrapper with temperature=%.2f", temp)
            wrapped = SoftmaxWrapper(model, temperature=temp)
            wrapped.eval()
            return wrapped
        except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
            logger.debug("SoftmaxWrapper unavailable or unnecessary: %s", exc)
    return model


def unwrap_export_model(model: nn.Module) -> nn.Module:
    """Return the wrapped module of a ``DataParallel``/DDP model, else the model itself."""
    inner = getattr(model, "module", None)
    return inner if isinstance(inner, nn.Module) else model


def model_device(model: nn.Module) -> torch.device:
    """Return the device of the model's first parameter (CPU for parameter-free models)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        logger.debug("Model has no parameters; exporting on the CPU")
        return torch.device("cpu")


@dataclass(frozen=True)
class OnnxGraphSpec:
    """Graph options for one ONNX export: I/O names, dynamic axes, opset and weight layout.

    ``external_data`` moves the weights into a ``<file>.data`` sidecar after the export; it is
    honoured by :func:`training.export.onnx.export_onnx`, not by :func:`run_onnx_export`.
    """

    input_names: list[str] = field(default_factory=lambda: ["input"])
    output_names: list[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: dict[str, dict[int, str]] | None = None
    opset: int = 17
    external_data: bool = False


def run_onnx_export(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    out_file: Path,
    graph: OnnxGraphSpec,
) -> None:
    """Run the TorchScript ``torch.onnx.export`` with the suite's standard options.

    The exporter's console chatter is captured and logged at debug level.
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            torch.onnx.export(
                model,
                inputs,
                str(out_file),
                export_params=True,
                opset_version=graph.opset,
                do_constant_folding=True,
                input_names=graph.input_names,
                output_names=graph.output_names,
                dynamic_axes=graph.dynamic_axes,
                dynamo=False,
            )
    finally:
        if buf.getvalue():
            logger.debug("torch.onnx.export output for %s:\n%s", out_file, buf.getvalue())


def detach_onnx_weights(onnx_path: Path, data_filename: str) -> None:
    """Move an ONNX model's tensors into one external ``data_filename`` sidecar file."""
    onnx_model = onnx.load(str(onnx_path))
    onnx.external_data_helper.convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(onnx_model, str(onnx_path))
