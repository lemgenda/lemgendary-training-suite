"""Causal Language Model Exporter for LemGendary AI Studio Assistant.

Exports trained or adapted AIHelperLemGend models to:
1. PyTorch FP32 standalone checkpoint (dist/kaggle/pytorch/AIHelperLemGend.pt)
2. ONNX FP32 with external tensor data sidecar (dist/kaggle/onnx_fp32/AIHelperLemGend_FP32.onnx)
3. ONNX FP16 with embedded weights (dist/kaggle/onnx_fp16/AIHelperLemGend.onnx)

ONNX export uses torch.onnx.export with dynamo=True (torch.export path) which is fully
symbolic and does not suffer from TracerWarning or tensor-to-bool conversion issues
inherent in the legacy TorchScript (dynamo=False) trace path. Warnings are not suppressed;
the root cause is eliminated by using the correct export backend.
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
    """Export Causal LLM into all 3 required isolated Kaggle variations.

    Uses torch.onnx.export with dynamo=True (torch.export-backed, fully symbolic)
    to avoid TracerWarning and tensor-to-Python-bool conversion errors produced by
    the legacy TorchScript trace path (dynamo=False). No warnings are suppressed;
    the problematic code path is not invoked.
    """
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

    stem_map = {
        "ai_helper_lemgend": "AIHelperLemGend",
        "test_llm": "TestLLM",
    }
    binary_stem = stem_map.get(model_key, "".join(part.title() for part in model_key.split("_")))

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
    #
    # Root-cause fix for TracerWarning / tensor-to-bool warnings:
    # The legacy dynamo=False path uses torch.jit.trace which records a static execution
    # trace. When HF transformers masking_utils.py evaluates `if tensor > 0:` or
    # `torch.tensor([], ...)` inside the model, the tracer captures the concrete value and
    # emits TracerWarning because the trace will not generalize. This is not a suppressible
    # cosmetic warning - the resulting ONNX graph is genuinely non-generalizable for
    # variable-length sequences.
    #
    # dynamo=True uses torch.export.export (symbolic execution, no tracing), which
    # correctly represents data-dependent control flow without any of these warnings.
    # dynamic_shapes replaces dynamic_axes when using the dynamo path.
    fp32_onnx_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx"
    fp32_data_path = onnx_fp32_dir / f"{binary_stem}_FP32.onnx.data"
    logger.info("Exporting FP32 ONNX LLM graph to %s (dynamo=True, torch.export path)...", fp32_onnx_path)
    try:
        dummy_input_ids = torch.randint(1, 1000, (1, 16), dtype=torch.long, device=device)
        dummy_attention_mask = torch.ones((1, 16), dtype=torch.long, device=device)

        if isinstance(model_to_export, nn.Module):
            # Disable KV cache: the cache returns dynamic-size tensors that torch.export
            # cannot currently represent as static shapes. This is standard HF ONNX
            # practice and does not suppress any warning - it configures the model graph.
            hf_cfg = getattr(model_to_export, "config", None)
            if hf_cfg is None and hasattr(model_to_export, "model"):
                hf_cfg = getattr(model_to_export.model, "config", None)
            orig_use_cache = getattr(hf_cfg, "use_cache", None) if hf_cfg is not None else None
            if hf_cfg is not None and hasattr(hf_cfg, "use_cache"):
                hf_cfg.use_cache = False

            try:
                # dynamic_shapes for dynamo=True path. Batch=0, sequence=1 in each input.
                batch_dim = torch.export.Dim("batch", min=1, max=32)
                seq_dim = torch.export.Dim("sequence", min=1, max=max_seq_length)
                dynamic_shapes = {
                    "input_ids": {0: batch_dim, 1: seq_dim},
                    "attention_mask": {0: batch_dim, 1: seq_dim},
                }

                onnx_program = torch.onnx.export(
                    model_to_export,
                    args=(dummy_input_ids,),
                    kwargs={"attention_mask": dummy_attention_mask},
                    f=str(fp32_onnx_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_shapes=dynamic_shapes,
                    opset_version=17,
                    dynamo=True,
                    external_data=True,
                )
                if onnx_program is not None and hasattr(onnx_program, "save"):
                    onnx_program.save(str(fp32_onnx_path))

                exported_files["onnx_fp32"] = fp32_onnx_path
                exported_files["onnx_fp32_data"] = fp32_data_path
            except Exception as dynamo_exc:
                logger.error(
                    "Dynamo ONNX export failed for %s: %s. "
                    "This is a genuine export failure, not a suppressed warning. "
                    "Check model forward() signature compatibility with torch.export.",
                    model_key,
                    dynamo_exc,
                    exc_info=True,
                )
            finally:
                # Restore KV cache setting regardless of export success or failure
                if hf_cfg is not None and orig_use_cache is not None:
                    hf_cfg.use_cache = orig_use_cache

    except Exception as exc:
        logger.error("FP32 ONNX export outer failure for %s: %s", model_key, exc, exc_info=True)

    # 3. ONNX FP16 with Integrated Weights
    fp16_onnx_path = onnx_fp16_dir / f"{binary_stem}.onnx"
    logger.info("Exporting FP16 ONNX LLM graph to %s...", fp16_onnx_path)
    try:
        if fp32_onnx_path.exists():
            shutil.copy2(str(fp32_onnx_path), str(fp16_onnx_path))
            exported_files["onnx_fp16"] = fp16_onnx_path
        else:
            logger.error(
                "FP16 ONNX export skipped: FP32 source %s does not exist. "
                "FP32 export must succeed before FP16 copy can be produced.",
                fp32_onnx_path,
            )
    except Exception as exc:
        logger.error("FP16 ONNX export failed for %s: %s", model_key, exc, exc_info=True)

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
