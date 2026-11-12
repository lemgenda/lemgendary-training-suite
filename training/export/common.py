"""Common export helpers, path resolution, and model wrapping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.nima import SoftmaxWrapper
from training.utils.model_payload import model_payload_of
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
        dims = list(size_raw)
        if len(dims) == 3:
            h, w = int(dims[1]), int(dims[2])
        elif len(dims) >= 2:
            h, w = int(dims[0]), int(dims[1])
        else:
            h = w = int(dims[0])
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


def load_checkpoint_weights(model: nn.Module, ckpt: Any) -> None:
    """Load a checkpoint's model state into ``model`` (non-strict, DDP ``module.`` stripped).

    ``ckpt`` is anything :func:`training.utils.model_payload.model_payload_of` accepts: a
    release payload, a checkpoint, a bare state dict or a module.
    """
    raw_state = model_payload_of(ckpt).state_dict
    state_dict = {k.removeprefix("module."): v for k, v in raw_state.items()}
    model.load_state_dict(state_dict, strict=False)


def model_device(model: nn.Module) -> torch.device:
    """Return the device of the model's first parameter (CPU for parameter-free models)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        logger.debug("Model has no parameters; exporting on the CPU")
        return torch.device("cpu")


DEFAULT_ONNX_OPSET = 18
"""Opset of every exported graph: the lowest opset the ``torch.export``-based exporter emits."""


@dataclass(frozen=True)
class OnnxGraphSpec:
    """Graph options for one ONNX export: I/O names, dynamic axes and opset.

    ``dynamic_axes`` maps an input name to ``{axis index: axis name}``; output axes are
    inferred by the exporter. Weight layout is a property of the variant, not of the graph
    (see :mod:`training.export.onnx_weights`).
    """

    input_names: list[str] = field(default_factory=lambda: ["input"])
    output_names: list[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: dict[str, dict[int, str]] | None = None
    opset: int = DEFAULT_ONNX_OPSET


def default_graph_spec(
    sample: torch.Tensor,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> OnnxGraphSpec:
    """Spec with dynamic batch, plus dynamic height and width for 4-D image inputs."""
    in_names = input_names or ["input"]
    out_names = output_names or ["output"]
    image_axes = {0: "batch", 2: "height", 3: "width"}
    default_axes = {in_names[0]: image_axes if sample.ndim >= 4 else {0: "batch"}}
    return OnnxGraphSpec(
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes or default_axes,
    )
