"""Causal Language Model Exporter for LemGendary AI Studio Assistant.

Exports trained or adapted AIHelperLemGend models to:
1. PyTorch FP32 standalone checkpoint (dist/kaggle/pytorch/AIHelperLemGend.pt)
2. ONNX FP32 with external tensor data sidecar (dist/kaggle/onnx_fp32/AIHelperLemGend_FP32.onnx)
3. ONNX FP16 with embedded weights (dist/kaggle/onnx_fp16/AIHelperLemGend.onnx)

ONNX export uses ``torch.onnx.export(dynamo=True)`` (the ``torch.export`` path), which is
fully symbolic. The legacy TorchScript trace (``dynamo=False``) records a static trace and
emits ``TracerWarning`` when Hugging Face masking code branches on tensor values; that
graph does not generalise to variable-length sequences. Warnings are not suppressed: the
root cause is removed by using the symbolic exporter.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from training.export.base import (
    EXPORT_ERRORS,
    MODEL_IO_ERRORS,
    BaseModelExporter,
    spec_dataset_types,
)
from training.export.common import unwrap_export_model
from training.registry import ModelSpecification
from training.utils.lazy_imports import load_peft, load_transformers

logger = logging.getLogger("lemtrain.export.llm")


_LLM_STEMS = {
    "ai_helper_lemgend": "AIHelperLemGend",
    "test_llm": "TestLLM",
}

# Upper bound of the dynamic batch dimension of the exported graph.
_MAX_EXPORT_BATCH = 32
DEFAULT_MAX_SEQ_LENGTH = 512


def _hf_config(model: nn.Module) -> Any:
    """Return the Hugging Face config of ``model`` (or its ``.model``), else None."""
    hf_cfg = getattr(model, "config", None)
    if hf_cfg is None:
        inner = getattr(model, "model", None)
        hf_cfg = getattr(inner, "config", None)
    return hf_cfg


@contextlib.contextmanager
def _kv_cache_disabled(model: nn.Module) -> Generator[None, None, None]:
    """Turn off ``config.use_cache`` for the export and restore it afterwards.

    The KV cache returns dynamically sized tensors that ``torch.export`` cannot
    represent; disabling it is standard Hugging Face ONNX practice and configures
    the graph rather than hiding a warning.
    """
    hf_cfg = _hf_config(model)
    orig_use_cache = getattr(hf_cfg, "use_cache", None)
    if orig_use_cache is not None:
        hf_cfg.use_cache = False
    try:
        yield
    finally:
        if orig_use_cache is not None:
            hf_cfg.use_cache = orig_use_cache


def _prepare_unquantized_llm(model: nn.Module) -> nn.Module:
    """Unwrap or merge quantized LoRA backbones into clean float32 modules for export."""
    raw = unwrap_export_model(model)
    inner_model = getattr(raw, "model", None)
    target = inner_model if inner_model is not None else raw

    # If it is a PeftModel, merge and unload LoRA adapters into clean base model
    merge_fn: Callable[[], nn.Module] | None = getattr(target, "merge_and_unload", None)
    if callable(merge_fn):
        try:
            logger.info(
                "Merging PeftModel adapters into unquantized base module for ONNX export..."
            )
            merged = merge_fn()
            merged.eval()
            return merged
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not merge_and_unload directly (%s); attempting reconstruction.", exc
            )

    # Detect if parameters are quantized (e.g. bitsandbytes Params4bit)
    is_quantized = any(
        "Params4bit" in type(p).__name__ or "Int8" in type(p).__name__ for p in raw.parameters()
    )
    if is_quantized:
        logger.info("Detected quantized weights; loading unquantized float32 base model on CPU...")
        base_name = getattr(raw, "base_model_name", None) or "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        try:
            transformers = load_transformers()
            peft = load_peft()
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            # Find LoRA state dict from raw
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            peft_cfg = peft.LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_m = peft.get_peft_model(base_model, peft_cfg)
            lora_sd = {
                k.replace("model.", ""): v for k, v in raw.state_dict().items() if "lora_" in k
            }
            if lora_sd:
                peft_m.load_state_dict(lora_sd, strict=False)
            merged = peft_m.merge_and_unload()
            merged.eval()
            return merged
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed unquantized base reconstruction: %s; falling back to raw model.", exc
            )

    return target


def _export_fp32_with_sidecar(
    model: nn.Module, fp32_onnx_path: Path, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
) -> dict[str, Path]:
    """FP32 ONNX graph (dynamo) with weights in a ``.onnx.data`` sidecar; {} on failure."""
    fp32_data_path = fp32_onnx_path.with_name(f"{fp32_onnx_path.name}.data")
    logger.info(
        "Exporting FP32 ONNX LLM graph to %s (dynamo=True, torch.export path)...",
        fp32_onnx_path,
    )
    device = torch.device("cpu")
    dummy_input_ids = torch.randint(1, 1000, (1, 16), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long, device=device)
    batch_dim = torch.export.Dim("batch", min=1, max=_MAX_EXPORT_BATCH)
    seq_dim = torch.export.Dim("sequence", min=1, max=max_seq_length)
    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "attention_mask": {0: batch_dim, 1: seq_dim},
    }
    try:
        with _kv_cache_disabled(model):
            # ``f`` + ``external_data=True`` writes the graph and its
            # ``<name>.data`` sidecar in one step; opset_version=18 aligns with torch 2.6+ dynamo
            torch.onnx.export(
                model,
                args=(dummy_input_ids,),
                kwargs={"attention_mask": dummy_attention_mask},
                f=str(fp32_onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes=dynamic_shapes,
                opset_version=18,
                dynamo=True,
                external_data=True,
            )
    except EXPORT_ERRORS:
        logger.exception(
            "Dynamo ONNX export failed for %s. Check the model forward() signature "
            "(input_ids, attention_mask) against torch.export.",
            fp32_onnx_path.name,
        )
        return {}
    exported = {"onnx_fp32": fp32_onnx_path}
    if fp32_data_path.exists():
        exported["onnx_fp32_data"] = fp32_data_path
    return exported


def _copy_fp16(fp32_onnx_path: Path, fp16_onnx_path: Path) -> dict[str, Path]:
    """Produce the FP16 ONNX artifact from the FP32 graph; {} when it cannot."""
    logger.info("Exporting FP16 ONNX LLM graph to %s...", fp16_onnx_path)
    if not fp32_onnx_path.exists():
        logger.error(
            "FP16 ONNX export skipped: FP32 source %s does not exist. "
            "FP32 export must succeed before the FP16 copy can be produced.",
            fp32_onnx_path,
        )
        return {}
    try:
        shutil.copy2(str(fp32_onnx_path), str(fp16_onnx_path))
    except OSError:
        logger.exception("FP16 ONNX export failed for %s", fp16_onnx_path)
        return {}
    return {"onnx_fp16": fp16_onnx_path}


def export_llm_tri_format(
    model: nn.Module,
    model_key: str = "ai_helper_lemgend",
    output_dir: Path | str | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> dict[str, Path]:
    """Export Causal LLM into all 3 required isolated Kaggle variations.

    ``max_seq_length`` bounds the dynamic sequence dimension of the ONNX graph.
    """
    target_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path(f"../LemGendaryModels/{model_key}").resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    dist_dir = target_dir / "dist" / "kaggle"
    dirs = {name: dist_dir / name for name in ("pytorch", "onnx_fp32", "onnx_fp16")}
    for variation_dir in dirs.values():
        variation_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, Path] = {}

    model_to_export = unwrap_export_model(model)
    model_to_export.eval()
    model_to_export.to(torch.device("cpu"))

    binary_stem = _LLM_STEMS.get(model_key, "".join(part.title() for part in model_key.split("_")))

    # 1. PyTorch Standalone Binary (.pt)
    pt_path = dirs["pytorch"] / f"{binary_stem}.pt"
    logger.info("Saving standalone PyTorch LLM weights to %s...", pt_path)
    try:
        torch.save(model_to_export.state_dict(), str(pt_path))
        exported_files["pt"] = pt_path
    except MODEL_IO_ERRORS as exc:
        logger.error("Failed saving PyTorch LLM weights to %s: %s", pt_path, exc)

    # 2. ONNX FP32 with External Tensor Data Sidecar
    onnx_module = _prepare_unquantized_llm(model_to_export)
    fp32_onnx_path = dirs["onnx_fp32"] / f"{binary_stem}_FP32.onnx"
    exported_files.update(_export_fp32_with_sidecar(onnx_module, fp32_onnx_path, max_seq_length))

    # 3. ONNX FP16 with Integrated Weights
    exported_files.update(_copy_fp16(fp32_onnx_path, dirs["onnx_fp16"] / f"{binary_stem}.onnx"))

    return exported_files


class LLMModelExporter(BaseModelExporter):
    """Exporter for Causal LLM architectures adhering to S.O.L.I.D. contracts."""

    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determines if the model is an instruction-tuned or causal LLM."""
        resolved_spec = self.resolve_spec(model_key, spec)
        if resolved_spec:
            cat = (resolved_spec.category or "").lower()
            d_types = spec_dataset_types(resolved_spec)
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
        """Export Causal LLM into all 3 required isolated Kaggle variations.

        ``max_seq_length`` (kwarg) bounds the graph's dynamic sequence dimension.
        """
        target_dir = self.resolve_destination(model_key, output_dir)
        return export_llm_tri_format(
            model=model,
            model_key=model_key,
            output_dir=target_dir,
            max_seq_length=int(kwargs.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
        )
