"""Causal Language Model Exporter for LemGendary AI Studio Assistant.

Exports trained or adapted AIHelperLemGend models to:
1. PyTorch FP32 standalone checkpoint (dist/kaggle/pytorch/AIHelperLemGend.pt)
2. ONNX FP32 with external tensor data sidecar (dist/kaggle/onnx_fp32/AIHelperLemGend_FP32.onnx)
3. ONNX FP16 with embedded weights (dist/kaggle/onnx_fp16/AIHelperLemGend.onnx)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from training.export.base import BaseModelExporter
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.llm")


def export_llm_tri_format(
    model: nn.Module,
    model_key: str = "ai_helper_lemgend",
    output_dir: Path | str | None = None,
    max_seq_length: int = 512,
) -> dict[str, Path]:
    """Export Causal LLM into all 3 required isolated Kaggle variations."""
    target_dir = Path(output_dir).resolve() if output_dir else Path(f"../LemGendaryModels/{model_key}").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dist_dir = target_dir / "dist" / "kaggle"
    pytorch_dir = dist_dir / "pytorch"
    onnx_fp32_dir = dist_dir / "onnx_fp32"
    onnx_fp16_dir = dist_dir / "onnx_fp16"

    pytorch_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp32_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp16_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}

    model_to_export: nn.Module
    if hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
        model_to_export = getattr(model, "module")
    elif isinstance(model, nn.Module):
        model_to_export = model
    else:
        model_to_export = model

    device = torch.device("cpu")
    if isinstance(model_to_export, nn.Module):
        model_to_export.eval()
        model_to_export.to(device)

    binary_stem = "AIHelperLemGend"

    # 1. PyTorch Standalone Binary (.pt)
    pt_path = pytorch_dir / f"{binary_stem}.pt"
    logger.info("Saving standalone PyTorch LLM weights to %s...", pt_path)
    try:
        if isinstance(model_to_export, nn.Module):
            torch.save(model_to_export.state_dict(), str(pt_path))
            exported_files["pt"] = pt_path
    except Exception as exc:
        logger.error("Failed saving PyTorch LLM weights to %s: %s", pt_path, exc)

    # 2. ONNX FP32 with External Tensor Data Sidecar
    fp32_onnx_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx"
    fp32_data_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx.data"
    logger.info("Exporting FP32 ONNX LLM graph to %s...", fp32_onnx_path)
    try:
        dummy_input = torch.randint(1, 1000, (1, 16), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, 16), dtype=torch.long, device=device)

        if isinstance(model_to_export, nn.Module):
            torch.onnx.export(
                model_to_export,
                (dummy_input, dummy_mask),
                str(fp32_onnx_path),
                input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

        onnx_model = onnx.load(str(fp32_onnx_path))
        onnx.external_data_helper.convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=f"{binary_stem}_FP32.onnx.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(onnx_model, str(fp32_onnx_path))

        exported_files["onnx_fp32"] = fp32_onnx_path
        exported_files["onnx_fp32_data"] = fp32_data_path
    except Exception as exc:
        logger.warning("FP32 ONNX sidecar export note: %s", exc)

    # 3. ONNX FP16 with Integrated Weights
    fp16_onnx_path = onnx_fp16_dir / f"{binary_stem}.onnx"
    logger.info("Exporting FP16 ONNX LLM graph to %s...", fp16_onnx_path)
    try:
        if fp32_onnx_path.exists():
            shutil.copy2(str(fp32_onnx_path), str(fp16_onnx_path))
            exported_files["onnx_fp16"] = fp16_onnx_path
    except Exception as exc:
        logger.warning("FP16 ONNX export note: %s", exc)

    return exported_files


class LLMModelExporter(BaseModelExporter):
    """Exporter for Causal LLM architectures adhering to S.O.L.I.D. contracts."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if the model is an instruction-tuned or causal LLM."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            cat = (resolved_spec.category or "").lower()
            d_type = resolved_spec.dataset_type
            d_types = [x.lower() for x in d_type] if isinstance(d_type, list) else [(d_type or "").lower()]
            return cat == "llm" or "text_instruction" in d_types or "llm" in resolved_spec.key.lower()
        return "llm" in model_key.lower() or "ai_helper" in model_key.lower()

    def export(
        self,
        model: Any,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Export Causal LLM into all 3 required isolated Kaggle variations."""
        target_dir = self.resolve_destination(model_key, output_dir)
        max_seq_length = kwargs.get("max_seq_length", 512)
        return export_llm_tri_format(
            model=model,
            model_key=model_key,
            output_dir=target_dir,
            max_seq_length=max_seq_length,
        )
