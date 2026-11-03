"""PyTorch model exporter implementation for standard deep learning architectures.

Exports models to:
1. PyTorch standalone state dict (.pt)
2. ONNX FP32 with external binary data sidecar (.onnx + .onnx.data)
3. ONNX FP16 with integrated self-contained weights (.onnx)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch

from training.export.base import EXPORT_ERRORS, MODEL_IO_ERRORS, BaseModelExporter
from training.export.common import (
    OnnxGraphSpec,
    detach_onnx_weights,
    extract_input_dimensions,
    model_device,
    run_onnx_export,
    unwrap_export_model,
)
from training.registry import ModelSpecification
from training.utils.lazy_imports import load_onnx_float16

logger = logging.getLogger("lemtrain.export.pytorch")


def _dummy_input(resolved_spec: ModelSpecification | None, kwargs: dict[str, Any]) -> torch.Tensor:
    """``dummy_input`` from kwargs, else a random image at the spec's input size."""
    dummy_input: torch.Tensor | None = kwargs.get("dummy_input")
    if dummy_input is None:
        raw_cfg = resolved_spec.raw_config if resolved_spec else {}
        h, w = extract_input_dimensions(raw_cfg, default_size=256)
        dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)
    return dummy_input


def _graph_spec(dummy_input: torch.Tensor, kwargs: dict[str, Any]) -> OnnxGraphSpec:
    """ONNX names, dynamic axes and opset from kwargs (image axes for 4-D inputs)."""
    opset: int = kwargs.get("opset", 17)
    in_names: list[str] = kwargs.get("input_names") or ["input"]
    out_names: list[str] = kwargs.get("output_names") or ["output"]
    default_axes = (
        {
            in_names[0]: {0: "batch", 2: "height", 3: "width"},
            out_names[0]: {0: "batch", 2: "height", 3: "width"},
        }
        if dummy_input.ndim >= 4
        else {
            in_names[0]: {0: "batch"},
            out_names[0]: {0: "batch"},
        }
    )
    dyn_axes: dict[str, dict[int, str]] = kwargs.get("dynamic_axes") or default_axes
    return OnnxGraphSpec(
        input_names=in_names, output_names=out_names, dynamic_axes=dyn_axes, opset=opset
    )


@dataclass
class _ExportJob:
    """One model's export into ``.pt``, FP32 ONNX (+ sidecar) and FP16 ONNX."""

    model: Any
    model_key: str
    target_dir: Path
    dummy_input: torch.Tensor
    graph: OnnxGraphSpec
    device: Any
    exported_files: dict[str, Path] = field(default_factory=dict)

    def _onnx_export(self, dtype: torch.dtype, path: Path) -> None:
        """Cast the model and the dummy input to ``dtype`` and export to ``path``."""
        dummy = self.dummy_input.to(dtype=dtype, device=self.device)
        self.model.to(dtype=dtype)
        run_onnx_export(self.model, (dummy,), path, self.graph)

    def save_pt(self) -> None:
        """1. FP32 PyTorch Model (.pt)."""
        pt_path = self.target_dir / f"{self.model_key}.pt"
        logger.info("Saving FP32 PyTorch model to %s...", pt_path)
        try:
            state_dict = (
                self.model.state_dict() if hasattr(self.model, "state_dict") else self.model
            )
            torch.save(state_dict, str(pt_path))
            self.exported_files["pt"] = pt_path
            logger.info("Successfully saved FP32 PyTorch model to %s", pt_path)
        except MODEL_IO_ERRORS as exc:
            logger.error("Failed saving FP32 PyTorch model to %s: %s", pt_path, exc)

    def export_fp32(self) -> None:
        """2. FP32 ONNX Model with Separate Weights Sidecar (.onnx + .onnx.data)."""
        fp32_onnx_path = self.target_dir / f"{self.model_key}_FP32.onnx"
        fp32_data_filename = f"{self.model_key}_FP32.onnx.data"
        fp32_data_path = self.target_dir / fp32_data_filename

        logger.info("Exporting FP32 ONNX model with external sidecar to %s...", fp32_onnx_path)
        try:
            self._onnx_export(torch.float32, fp32_onnx_path)
            detach_onnx_weights(fp32_onnx_path, fp32_data_filename)

            self.exported_files["onnx_fp32"] = fp32_onnx_path
            self.exported_files["onnx_fp32_data"] = fp32_data_path
            logger.info("Successfully exported FP32 ONNX with sidecar: %s", fp32_onnx_path.name)
        except EXPORT_ERRORS as exc:
            logger.error("Failed exporting FP32 ONNX sidecar for %s: %s", self.model_key, exc)

    def export_fp16(self) -> None:
        """3. FP16 ONNX Model with Integrated Weights (.onnx); converter fallback."""
        fp16_onnx_path = self.target_dir / f"{self.model_key}.onnx"
        logger.info("Exporting FP16 integrated ONNX model to %s...", fp16_onnx_path)
        try:
            self._onnx_export(torch.float16, fp16_onnx_path)

            self.exported_files["onnx_fp16"] = fp16_onnx_path
            logger.info("Successfully exported FP16 integrated ONNX: %s", fp16_onnx_path.name)
        except EXPORT_ERRORS as exc:
            logger.warning(
                "Direct FP16 torch.onnx.export failed for %s (%s). Falling back to "
                "onnxconverter...",
                self.model_key,
                exc,
            )
            self._convert_fp16(fp16_onnx_path)
        finally:
            self.model.to(dtype=torch.float32)

    def _convert_fp16(self, fp16_onnx_path: Path) -> None:
        """Export FP32 to a temporary file and convert it to FP16 with onnxconverter."""
        try:
            tmp_fp32 = self.target_dir / f"{self.model_key}_temp_fp32.onnx"
            self.model.to(dtype=torch.float32)
            dummy_fp32 = self.dummy_input.to(dtype=torch.float32, device=self.device)
            run_onnx_export(self.model, (dummy_fp32,), tmp_fp32, self.graph)
            float16 = load_onnx_float16()
            m_fp32 = onnx.load(str(tmp_fp32))
            m_fp16 = float16.convert_float_to_float16(m_fp32)
            onnx.save(m_fp16, str(fp16_onnx_path))
            if tmp_fp32.exists():
                tmp_fp32.unlink()
            self.exported_files["onnx_fp16"] = fp16_onnx_path
            logger.info(
                "Successfully converted and saved FP16 ONNX model via onnxconverter fallback to %s",
                fp16_onnx_path,
            )
        except EXPORT_ERRORS as fallback_exc:
            logger.error("Failed FP16 fallback export for %s: %s", self.model_key, fallback_exc)


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
        """Export PyTorch model into all 3 required formats."""
        target_dir = self.resolve_destination(model_key, output_dir)
        resolved_spec = self.resolve_spec(model_key, spec)

        dummy_input = _dummy_input(resolved_spec, kwargs)
        graph = _graph_spec(dummy_input, kwargs)
        model_to_export = unwrap_export_model(model)
        model_to_export.eval()
        job = _ExportJob(
            model=model_to_export,
            model_key=model_key,
            target_dir=target_dir,
            dummy_input=dummy_input,
            graph=graph,
            device=model_device(model_to_export),
        )

        job.save_pt()
        job.export_fp32()
        job.export_fp16()
        return job.exported_files
