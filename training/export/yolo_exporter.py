"""Ultralytics YOLO model exporter implementation.

Exports YOLOv8n object detection model into:
1. PyTorch standalone model (.pt)
2. ONNX FP32 with external data sidecar (.onnx + .onnx.data)
3. ONNX FP16 integrated weights (.onnx; sidecar only above 2 GB)

The ONNX graph comes from Ultralytics' own exporter, not from this suite's
``torch.export`` path, because it carries the detection head post-processing.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import onnx

from training.cloud.variations import PT_SUFFIX
from training.export.base import EXPORT_ERRORS, BaseModelExporter
from training.export.onnx_variants import (
    ONNX_FP16_VARIANT,
    ONNX_FP32_VARIANT,
    convert_proto_to_fp16,
)
from training.export.onnx_weights import save_proto
from training.registry import ModelSpecification
from training.utils.lazy_imports import load_ultralytics

logger = logging.getLogger("lemtrain.export.yolo")


def _mirror_pt(model: Any, pt_path: Path, exported_files: dict[str, Path]) -> None:
    """1. FP32 PyTorch model (.pt): mirror ``model.best`` or keep an existing ``pt_path``."""
    try:
        best_pt = getattr(model, "best", None)
        if best_pt and Path(best_pt).exists():
            shutil.copy2(str(best_pt), str(pt_path))
            exported_files["pt"] = pt_path
            logger.info("Mirrored YOLO FP32 PyTorch model to %s", pt_path)
        elif pt_path.exists():
            exported_files["pt"] = pt_path
    except (OSError, TypeError) as exc:
        logger.error("Failed saving YOLO .pt to %s: %s", pt_path, exc)


def _load_yolo(weights: Path | str) -> Any:
    """Load an Ultralytics YOLO model from ``weights`` (None when ultralytics is missing)."""
    try:
        return load_ultralytics().YOLO(str(weights))
    except ImportError:
        logger.warning("ultralytics package not installed for YOLO export.")
        return None


def _resolve_yolo_model(model: Any, pt_path: Path) -> Any:
    """The exportable YOLO model: ``model`` itself, else loaded from ``model.best`` or .pt."""
    if hasattr(model, "export"):
        return model
    if hasattr(model, "best") and Path(getattr(model, "best", "")).exists():
        return _load_yolo(model.best)
    if pt_path.exists():
        return _load_yolo(pt_path)
    return None


def _ultralytics_onnx(yolo_model: Any) -> Path | None:
    """FP32 ONNX graph written by Ultralytics' own exporter (its detection head included)."""
    if yolo_model is None or not hasattr(yolo_model, "export"):
        return None
    raw_onnx = yolo_model.export(format="onnx", half=False, imgsz=640)
    return Path(raw_onnx) if raw_onnx and Path(raw_onnx).exists() else None


def _export_onnx_variants(
    yolo_model: Any, stem: str, target_dir: Path, exported_files: dict[str, Path]
) -> None:
    """2. FP32 ONNX + ``.onnx.data`` sidecar and 3. FP16 ONNX (embedded below 2 GB).

    Ultralytics only writes FP16 graphs on a CUDA device, so the FP16 variant is converted
    from the FP32 graph; both files go through the shared weight-layout writer.
    """
    try:
        raw_onnx = _ultralytics_onnx(yolo_model)
        if raw_onnx is None:
            return
        fp16_proto = convert_proto_to_fp16(onnx.load(str(raw_onnx)), keep_io_types=False)
        fp32_path = target_dir / ONNX_FP32_VARIANT.file_name(stem)
        fp32_data = save_proto(onnx.load(str(raw_onnx)), fp32_path, ONNX_FP32_VARIANT.layout)
        exported_files[ONNX_FP32_VARIANT.key] = fp32_path
        if fp32_data is not None:
            exported_files[ONNX_FP32_VARIANT.data_key] = fp32_data
        fp16_path = target_dir / ONNX_FP16_VARIANT.file_name(stem)
        fp16_data = save_proto(fp16_proto, fp16_path, ONNX_FP16_VARIANT.layout)
        exported_files[ONNX_FP16_VARIANT.key] = fp16_path
        if fp16_data is not None:
            exported_files[ONNX_FP16_VARIANT.data_key] = fp16_data
        logger.info("Exported YOLO ONNX variants %s and %s", fp32_path.name, fp16_path.name)
    except EXPORT_ERRORS as exc:
        logger.error("Failed exporting YOLO ONNX variants: %s", exc)


class YOLOModelExporter(BaseModelExporter):
    """Exports Ultralytics YOLO models adhering to canonical tri-format rules."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if the model is an Ultralytics YOLO architecture."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            return (
                resolved_spec.key == "yolov8n"
                or resolved_spec.class_name == "YOLOv8n"
                or "yolo" in resolved_spec.key.lower()
            )
        return "yolo" in model_key.lower()

    def export(
        self,
        model: Any,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Export Ultralytics YOLO model into all 3 required formats."""
        target_dir = self.resolve_destination(model_key, output_dir)
        exported_files: dict[str, Path] = {}

        pt_path = target_dir / f"{model_key}{PT_SUFFIX}"
        _mirror_pt(model, pt_path, exported_files)
        yolo_model = _resolve_yolo_model(model, pt_path)
        _export_onnx_variants(yolo_model, model_key, target_dir, exported_files)
        return exported_files
