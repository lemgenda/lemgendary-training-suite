"""PyTorch model exporter implementation for standard deep learning architectures.

Exports models to:
1. PyTorch standalone state dict (.pt)
2. ONNX FP32 with external binary data sidecar (.onnx + .onnx.data)
3. ONNX FP16 with integrated self-contained weights (.onnx)
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from training.export.base import BaseModelExporter
from training.export.common import extract_input_dimensions
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.pytorch")


class PyTorchModelExporter(BaseModelExporter):
    """Exports vision, restoration, aesthetic, and standard PyTorch models to tri-format."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """PyTorch exporter handles non-YOLO, non-LLM, non-forex models."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if not resolved_spec:
            return model_key != "yolov8n" and "forex" not in model_key.lower() and "ai_helper" not in model_key.lower()

        cat = str(resolved_spec.category or "").lower()
        d_type = resolved_spec.dataset_type
        d_types = [str(x).lower() for x in d_type] if isinstance(d_type, list) else [str(d_type or "").lower()]
        if cat in ("llm",) or "text_instruction" in d_types or "forex" in d_types:
            return False
        if resolved_spec.key == "yolov8n" or resolved_spec.class_name == "YOLOv8n":
            return False
        return True

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

        # Determine dummy input dimensions
        dummy_input: torch.Tensor | None = kwargs.get("dummy_input")
        if dummy_input is None:
            raw_cfg = resolved_spec.raw_config if resolved_spec else {}
            h, w = extract_input_dimensions(raw_cfg, default_size=256)
            dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)

        opset: int = kwargs.get("opset", 17)
        in_names: list[str] = kwargs.get("input_names") or ["input"]
        out_names: list[str] = kwargs.get("output_names") or ["output"]
        dyn_axes: dict[str, dict[int, str]] = kwargs.get("dynamic_axes") or {
            in_names[0]: {0: "batch", 2: "height", 3: "width"},
            out_names[0]: {0: "batch", 2: "height", 3: "width"},
        }

        model_to_export = model.module if hasattr(model, "module") else model
        if isinstance(model_to_export, nn.Module):
            model_to_export.eval()
            device = next(model_to_export.parameters()).device
        else:
            device = torch.device("cpu")

        exported_files: dict[str, Path] = {}

        # -------------------------------------------------------------------------
        # 1. FP32 PyTorch Model (.pt)
        # -------------------------------------------------------------------------
        pt_path = target_dir / f"{model_key}.pt"
        logger.info("Saving FP32 PyTorch model to %s...", pt_path)
        try:
            state_dict = model_to_export.state_dict() if hasattr(model_to_export, "state_dict") else model_to_export
            torch.save(state_dict, str(pt_path))
            exported_files["pt"] = pt_path
            logger.info("Successfully saved FP32 PyTorch model to %s", pt_path)
        except Exception as exc:
            logger.error("Failed saving FP32 PyTorch model to %s: %s", pt_path, exc)

        # -------------------------------------------------------------------------
        # 2. FP32 ONNX Model with Separate Weights Sidecar (.onnx + .onnx.data)
        # -------------------------------------------------------------------------
        fp32_onnx_path = target_dir / f"{model_key}_FP32.onnx"
        fp32_data_filename = f"{model_key}_FP32.onnx.data"
        fp32_data_path = target_dir / fp32_data_filename

        logger.info("Exporting FP32 ONNX model with external sidecar to %s...", fp32_onnx_path)
        buf_fp32 = io.StringIO()
        try:
            dummy_fp32 = dummy_input.to(dtype=torch.float32, device=device)
            if isinstance(model_to_export, nn.Module):
                model_to_export.to(dtype=torch.float32)

            with redirect_stdout(buf_fp32), redirect_stderr(buf_fp32):
                torch.onnx.export(
                    model_to_export,
                    (dummy_fp32,),
                    str(fp32_onnx_path),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dyn_axes,
                )

            # Convert to external data sidecar
            onnx_model = onnx.load(str(fp32_onnx_path))
            onnx.external_data_helper.convert_model_to_external_data(
                onnx_model,
                all_tensors_to_one_file=True,
                location=fp32_data_filename,
                size_threshold=1024,
                convert_attribute=False,
            )
            onnx.save(onnx_model, str(fp32_onnx_path))

            exported_files["onnx_fp32"] = fp32_onnx_path
            exported_files["onnx_fp32_data"] = fp32_data_path
            logger.info("Successfully exported FP32 ONNX with sidecar: %s", fp32_onnx_path.name)
        except Exception as exc:
            logger.error("Failed exporting FP32 ONNX sidecar for %s: %s", model_key, exc)

        # -------------------------------------------------------------------------
        # 3. FP16 ONNX Model with Integrated Weights (.onnx)
        # -------------------------------------------------------------------------
        fp16_onnx_path = target_dir / f"{model_key}.onnx"
        logger.info("Exporting FP16 integrated ONNX model to %s...", fp16_onnx_path)
        buf_fp16 = io.StringIO()
        try:
            dummy_fp16 = dummy_input.to(dtype=torch.float16, device=device)
            if isinstance(model_to_export, nn.Module):
                model_to_export.to(dtype=torch.float16)

            with redirect_stdout(buf_fp16), redirect_stderr(buf_fp16):
                torch.onnx.export(
                    model_to_export,
                    (dummy_fp16,),
                    str(fp16_onnx_path),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dyn_axes,
                )

            exported_files["onnx_fp16"] = fp16_onnx_path
            logger.info("Successfully exported FP16 integrated ONNX: %s", fp16_onnx_path.name)
        except Exception as exc:
            logger.warning(
                "Direct FP16 torch.onnx.export failed for %s (%s). Falling back to onnxconverter...",
                model_key,
                exc,
            )
            try:
                tmp_fp32 = target_dir / f"{model_key}_temp_fp32.onnx"
                if isinstance(model_to_export, nn.Module):
                    model_to_export.to(dtype=torch.float32)
                dummy_fp32 = dummy_input.to(dtype=torch.float32, device=device)
                torch.onnx.export(
                    model_to_export,
                    (dummy_fp32,),
                    str(tmp_fp32),
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dyn_axes,
                )
                from onnxconverter_common import float16

                m_fp32 = onnx.load(str(tmp_fp32))
                m_fp16 = float16.convert_float_to_float16(m_fp32)
                onnx.save(m_fp16, str(fp16_onnx_path))
                if tmp_fp32.exists():
                    tmp_fp32.unlink()
                exported_files["onnx_fp16"] = fp16_onnx_path
                logger.info(
                    "Successfully converted and saved FP16 ONNX model via onnxconverter fallback to %s",
                    fp16_onnx_path,
                )
            except Exception as fallback_exc:
                logger.error("Failed FP16 fallback export for %s: %s", model_key, fallback_exc)
        finally:
            if isinstance(model_to_export, nn.Module):
                model_to_export.to(dtype=torch.float32)

        return exported_files
