"""PyTorch model exporter implementation for standard deep learning architectures.

Exports models to the three release variants, named after the model's canonical stem:
1. ``<Stem>.pt``: self-contained PyTorch FP32 model
2. ``<Stem>_FP32.onnx`` + ``<Stem>_FP32.onnx.data``: ONNX FP32 with external weights sidecar
3. ``<Stem>.onnx``: ONNX FP16 with embedded weights (sidecar only above 2 GB)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from training.cloud.variations import PT_SUFFIX
from training.export.base import MODEL_IO_ERRORS, BaseModelExporter
from training.export.common import (
    ExportError,
    default_graph_spec,
    extract_input_dimensions,
    model_device,
    unwrap_export_model,
)
from training.export.onnx_variants import OnnxVariantWriter, write_release_variants
from training.export.torch_standalone import export_torch_standalone
from training.export.validation import resolve_model_stem
from training.registry import ModelSpecification
from training.registry.class_names import class_names
from training.registry.preprocessing import image_preprocessing

logger = logging.getLogger("lemtrain.export.pytorch")

EXPORT_SAMPLE_BATCH = 2
"""Batch of the export sample: ``torch.export`` specialises an axis whose sample size is 1."""


def _dummy_input(resolved_spec: ModelSpecification | None, kwargs: dict[str, Any]) -> torch.Tensor:
    """``dummy_input`` from kwargs, else a random image batch at the spec's input size."""
    dummy_input: torch.Tensor | None = kwargs.get("dummy_input")
    if dummy_input is None:
        raw_cfg = resolved_spec.raw_config if resolved_spec else {}
        h, w = extract_input_dimensions(raw_cfg, default_size=256)
        dummy_input = torch.randn(EXPORT_SAMPLE_BATCH, 3, h, w, dtype=torch.float32)
    elif dummy_input.ndim >= 1 and dummy_input.shape[0] == 1:
        dummy_input = dummy_input.expand(EXPORT_SAMPLE_BATCH, *dummy_input.shape[1:]).clone()
    return dummy_input


def pt_metadata(model_key: str, spec: ModelSpecification | None) -> dict[str, Any]:
    """``.pt`` metadata: model key, image preprocessing and a classifier's ``class_names``.

    The names map each logit index of the output to its class without the registry; the
    preprocessing (``training.registry.preprocessing``) is the one the model was trained with.
    """
    raw_config = spec.raw_config if spec else None
    metadata: dict[str, Any] = {
        "model_key": model_key,
        "preprocessing": image_preprocessing(raw_config).metadata(),
    }
    names = class_names(raw_config)
    if names:
        metadata["class_names"] = list(names)
    return metadata


def _save_pt(model: torch.nn.Module, pt_path: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    """1. Self-contained FP32 PyTorch model (``.pt``) with ``metadata``; {} on failure."""
    try:
        export_torch_standalone(model, pt_path, metadata=metadata)
    except (ExportError, *MODEL_IO_ERRORS) as exc:
        logger.error("Failed saving FP32 PyTorch model to %s: %s", pt_path, exc)
        return {}
    return {"pt": pt_path}


class PyTorchModelExporter(BaseModelExporter):
    """Exports vision, restoration, aesthetic, and standard PyTorch models to tri-format."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """PyTorch exporter handles non-YOLO, non-LLM, non-forex models."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if not resolved_spec:
            return (
                model_key != "yolov8n"
                and "forex" not in model_key.lower()
                and "ai_helper" not in model_key.lower()
            )

        cat = (resolved_spec.category or "").lower()
        d_type = resolved_spec.dataset_type
        d_types = (
            [x.lower() for x in d_type] if isinstance(d_type, list) else [(d_type or "").lower()]
        )
        if cat in ("llm",) or "text_instruction" in d_types or "forex" in d_types:
            return False
        return not (resolved_spec.key == "yolov8n" or resolved_spec.class_name == "YOLOv8n")

    def export(
        self,
        model: Any,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Export PyTorch model into all 3 required formats.

        ``dummy_input`` (kwarg) is the export sample; ``input_names``, ``output_names`` and
        ``dynamic_axes`` override the default image graph spec.
        """
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)
        stem = resolve_model_stem(model_key, resolved_spec)

        model_to_export = unwrap_export_model(model)
        model_to_export.eval()
        dummy_input = _dummy_input(resolved_spec, kwargs).to(model_device(model_to_export))
        graph = default_graph_spec(
            dummy_input,
            input_names=kwargs.get("input_names"),
            output_names=kwargs.get("output_names"),
            dynamic_axes=kwargs.get("dynamic_axes"),
        )
        exported = _save_pt(
            model_to_export,
            target_dir / f"{stem}{PT_SUFFIX}",
            pt_metadata(model_key, resolved_spec),
        )
        writer = OnnxVariantWriter(model_to_export, (dummy_input,), graph)
        exported.update(write_release_variants(writer, target_dir, stem))
        return exported
