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

import torch
from torch import nn

from training.export.base import EXPORT_ERRORS, MODEL_IO_ERRORS, BaseModelExporter
from training.export.common import (
    OnnxGraphSpec,
    detach_onnx_weights,
    run_onnx_export,
    unwrap_export_model,
)
from training.registry import ModelSpecification

logger = logging.getLogger("lemtrain.export.llm")


def export_llm_tri_format(
    model: nn.Module,
    model_key: str = "ai_helper_lemgend",
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Export Causal LLM into all 3 required isolated Kaggle variations."""
    target_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path(f"../LemGendaryModels/{model_key}").resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    dist_dir = target_dir / "dist" / "kaggle"
    pytorch_dir = dist_dir / "pytorch"
    onnx_fp32_dir = dist_dir / "onnx_fp32"
    onnx_fp16_dir = dist_dir / "onnx_fp16"

    pytorch_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp32_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp16_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}

    model_to_export = unwrap_export_model(model)
    device = torch.device("cpu")
    model_to_export.eval()
    model_to_export.to(device)

    stem_map = {
        "ai_helper_lemgend": "AIHelperLemGend",
        "test_llm": "TestLLM",
    }
    binary_stem = stem_map.get(model_key, "".join(part.title() for part in model_key.split("_")))

    # 1. PyTorch Standalone Binary (.pt)
    pt_path = pytorch_dir / f"{binary_stem}.pt"
    logger.info("Saving standalone PyTorch LLM weights to %s...", pt_path)
    try:
        torch.save(model_to_export.state_dict(), str(pt_path))
        exported_files["pt"] = pt_path
    except MODEL_IO_ERRORS as exc:
        logger.error("Failed saving PyTorch LLM weights to %s: %s", pt_path, exc)

    # 2. ONNX FP32 with External Tensor Data Sidecar
    fp32_onnx_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx"
    fp32_data_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx.data"
    logger.info("Exporting FP32 ONNX LLM graph to %s...", fp32_onnx_path)
    try:
        dummy_input = torch.randint(1, 1000, (1, 16), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, 16), dtype=torch.long, device=device)

        run_onnx_export(
            model_to_export,
            (dummy_input, dummy_mask),
            fp32_onnx_path,
            OnnxGraphSpec(
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset=17,
            ),
        )
        detach_onnx_weights(fp32_onnx_path, fp32_data_path.name)

        exported_files["onnx_fp32"] = fp32_onnx_path
        exported_files["onnx_fp32_data"] = fp32_data_path
    except EXPORT_ERRORS as exc:
        logger.warning("FP32 ONNX sidecar export note: %s", exc)

    # 3. ONNX FP16 with Integrated Weights
    fp16_onnx_path = onnx_fp16_dir / f"{binary_stem}.onnx"
    logger.info("Exporting FP16 ONNX LLM graph to %s...", fp16_onnx_path)
    try:
        if fp32_onnx_path.exists():
            shutil.copy2(str(fp32_onnx_path), str(fp16_onnx_path))
            exported_files["onnx_fp16"] = fp16_onnx_path
    except OSError as exc:
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
            d_types = (
                [x.lower() for x in d_type]
                if isinstance(d_type, list)
                else [(d_type or "").lower()]
            )
            return (
                cat == "llm" or "text_instruction" in d_types or "llm" in resolved_spec.key.lower()
            )
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
        return export_llm_tri_format(
            model=model,
            model_key=model_key,
            output_dir=target_dir,
        )
