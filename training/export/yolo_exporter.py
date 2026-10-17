"""Ultralytics YOLO model exporter implementation.

Exports YOLOv8n object detection model into:
1. PyTorch standalone model (.pt)
2. ONNX FP32 with external data sidecar (.onnx + .onnx.data)
3. ONNX FP16 integrated weights (.onnx)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import onnx

from training.export.base import BaseModelExporter
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.yolo")


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

        # -------------------------------------------------------------------------
        # 1. FP32 PyTorch Model (.pt)
        # -------------------------------------------------------------------------
        pt_path = target_dir / f"{model_key}.pt"
        try:
            best_pt = getattr(model, "best", None)
            if best_pt and Path(best_pt).exists():
                shutil.copy2(str(best_pt), str(pt_path))
                exported_files["pt"] = pt_path
                logger.info("Mirrored YOLO FP32 PyTorch model to %s", pt_path)
            elif pt_path.exists():
                exported_files["pt"] = pt_path
        except Exception as exc:
            logger.error("Failed saving YOLO .pt to %s: %s", pt_path, exc)

        yolo_model = None
        if hasattr(model, "export"):
            yolo_model = model
        elif hasattr(model, "best") and Path(getattr(model, "best", "")).exists():
            try:
                from ultralytics import YOLO

                yolo_model = YOLO(str(model.best))
            except ImportError:
                logger.warning("ultralytics package not installed for YOLO export.")
        elif pt_path.exists():
            try:
                from ultralytics import YOLO

                yolo_model = YOLO(str(pt_path))
            except ImportError:
                logger.warning("ultralytics package not installed for YOLO export.")

        # -------------------------------------------------------------------------
        # 2. FP32 ONNX with Separate Weights Sidecar (.onnx + .onnx.data)
        # -------------------------------------------------------------------------
        fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
        fp32_data_path = target_dir / f"{model_key}_FP32.onnx.data"
        try:
            if yolo_model is not None and hasattr(yolo_model, "export"):
                raw_onnx = yolo_model.export(format="onnx", half=False, imgsz=640)
                if raw_onnx and Path(raw_onnx).exists():
                    onnx_model = onnx.load(str(raw_onnx))
                    onnx.external_data_helper.convert_model_to_external_data(
                        onnx_model,
                        all_tensors_to_one_file=True,
                        location=f"{model_key}_FP32.onnx.data",
                        size_threshold=1024,
                        convert_attribute=False,
                    )
                    onnx.save(onnx_model, str(fp32_onnx_path))
                    exported_files["onnx_fp32"] = fp32_onnx_path
                    exported_files["onnx_fp32_data"] = fp32_data_path
                    logger.info("Successfully exported YOLO FP32 ONNX with sidecar to %s", fp32_onnx_path)
        except Exception as exc:
            logger.error("Failed exporting YOLO FP32 ONNX sidecar: %s", exc)

        # -------------------------------------------------------------------------
        # 3. FP16 ONNX with Integrated Weights (.onnx)
        # -------------------------------------------------------------------------
        fp16_onnx_path = target_dir / f"{model_key}.onnx"
        try:
            if yolo_model is not None and hasattr(yolo_model, "export"):
                raw_half_onnx = yolo_model.export(format="onnx", half=True, imgsz=640)
                if raw_half_onnx and Path(raw_half_onnx).exists():
                    if Path(raw_half_onnx).resolve() != fp16_onnx_path.resolve():
                        shutil.copy2(str(raw_half_onnx), str(fp16_onnx_path))
                    exported_files["onnx_fp16"] = fp16_onnx_path
                    logger.info("Successfully exported YOLO FP16 integrated ONNX to %s", fp16_onnx_path)
        except Exception as exc:
            logger.error("Failed exporting YOLO FP16 ONNX: %s", exc)

        # Clean up any misplaced/orphaned {model_key}.onnx.data
        orphaned_data = target_dir / f"{model_key}.onnx.data"
        if orphaned_data.exists():
            try:
                orphaned_data.unlink()
                logger.info("Cleaned up orphaned sidecar data %s", orphaned_data)
            except OSError:
                pass

        return exported_files
