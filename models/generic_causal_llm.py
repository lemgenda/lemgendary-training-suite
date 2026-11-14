"""Universal Generic Causal Language Model Fine-Tuning Wrapper.

Supports any HuggingFace Causal LM specified via base_foundation_model in
unified_models_v2.yaml (e.g. Qwen, Llama, Mistral, DeepSeek, Gemma),
automatic 4-bit / 8-bit quantization shielding via BitsAndBytesConfig,
and dynamic LoRA / PEFT adapter injection according to curriculum policies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.llm_chunked_loss import ChunkedCausalLMLoss, chunked_eval_loss_and_preds
from models.model_config import resolve_config
from training.utils.lazy_imports import load_peft, load_transformers

logger = logging.getLogger("lemtrain.models.generic_llm")


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_tokenizer(transformers: Any, name: str) -> Any:
    """Right-padding tokenizer for ``name``; falls back to EOS as the pad token."""
    raw_tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
        padding_side="right",
    )
    if raw_tokenizer is not None:
        pad_token = getattr(raw_tokenizer, "pad_token", None)
        if pad_token is None:
            eos_token = getattr(raw_tokenizer, "eos_token", None)
            raw_tokenizer.pad_token = eos_token
    return raw_tokenizer


def _quant_config(transformers: Any, load_in_4bit: bool) -> Any:
    """NF4 4-bit ``BitsAndBytesConfig`` when requested and available, else ``None``."""
    if not load_in_4bit:
        return None
    try:
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        logger.warning("bitsandbytes not installed; loading without 4-bit quantization.")
        return None


@dataclass(frozen=True)
class LoraSettings:
    """LoRA adapter and quantization settings of a ``GenericCausalLLM``.

    The registry ``model_kwargs`` stored with each checkpoint carry these fields
    as flat keywords.
    """

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True


class GenericCausalLLM(nn.Module):
    """Universal Causal Language Model fine-tuning backbone wrapper."""

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        *,
        lora: LoraSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Load ``base_model_name_or_path`` with ``lora`` (or its fields as flat keywords)."""
        super().__init__()
        settings = resolve_config(LoraSettings, lora, kwargs)
        self.base_model_name = base_model_name_or_path
        self._chunk_cfg = {
            "use_lora": settings.use_lora,
            "enabled": bool(kwargs.get("chunked_cross_entropy", True)),
            "chunk_size": int(kwargs.get("ce_chunk_size", 256)),
        }
        self.model: Any = None
        self.tokenizer: Any = None

        is_tpu = os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or bool(
            os.environ.get("TPU_NAME") or os.environ.get("KAGGLE_TPU_NAME")
        )
        effective_4bit = False if is_tpu else settings.load_in_4bit

        logger.info("Initializing GenericCausalLLM backbone: %s", self.base_model_name)

        if self.base_model_name == "mock" or kwargs.get("mock_mode", False):
            self._init_mock_model()
            return

        try:
            transformers = load_transformers()

            self.tokenizer = _load_tokenizer(transformers, self.base_model_name)
            if is_tpu:
                model_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                model_dtype = torch.float32
            model_kwargs: dict[str, Any] = {
                "quantization_config": _quant_config(transformers, effective_4bit),
                "torch_dtype": model_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if torch.cuda.is_available():
                model_kwargs["attn_implementation"] = "sdpa"

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs,
            )

            if self._chunk_cfg["use_lora"]:
                lora_cfg = {
                    "r": settings.lora_r,
                    "lora_alpha": settings.lora_alpha,
                    "lora_dropout": settings.lora_dropout,
                }
                self._inject_lora(
                    lora_cfg, settings.load_in_4bit, kwargs.get("target_modules", LORA_TARGETS)
                )

        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            # ImportError: transformers missing; OSError: weights/tokenizer not found or not
            # downloadable; ValueError: unsupported config; RuntimeError: backend/device failure.
            logger.warning(
                "Could not initialize full transformers model (%s). "
                "Using lightweight mock for testing.",
                exc,
            )
            self._init_mock_model()

    @property
    def head_param_prefixes(self) -> tuple[str, ...]:
        """The trainable ``lm_head`` output projection; empty under LoRA.

        A full fine-tune gives the vocabulary projection its own head LR. With LoRA the
        base model (``lm_head`` included) is frozen and the low-rank adapters train as one
        group, so there is no separate head; a tied ``lm_head`` is reported under its
        embedding name and is not split off either.
        """
        prefixes: list[str] = []
        for name, param in self.named_parameters():
            if not param.requires_grad or "lm_head" not in name.split("."):
                continue
            prefix = name[: name.index("lm_head") + len("lm_head")] + "."
            if prefix not in prefixes:
                prefixes.append(prefix)
        return tuple(prefixes)

    def _inject_lora(self, lora: dict, load_in_4bit: bool, target_modules: list) -> None:
        """Wrap ``self.model`` in a LoRA adapter (``lora``: r, lora_alpha, lora_dropout)."""
        try:
            peft = load_peft()

            if load_in_4bit:
                self.model = peft.prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=True
                )

            peft_config = peft.LoraConfig(
                **lora,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = peft.get_peft_model(self.model, peft_config)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            logger.info("Injected LoRA adapter into GenericCausalLLM backbone (r=%d)", lora["r"])
        except ImportError:
            logger.warning("PEFT not installed. GenericCausalLLM operating in standard mode.")

    def _init_mock_model(self) -> None:
        """Lightweight mock embedding and linear projection for testing and verification."""
        vocab_size = 32000
        hidden_dim = 256
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()

    def _chunked_forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        labels: Any,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor] | None:
        base_decoder: Any = getattr(self.model, "model", None) or getattr(
            getattr(self.model, "base_model", None), "model", None
        )
        lm_head = self.model.get_output_embeddings()
        if not isinstance(base_decoder, nn.Module) or lm_head is None:
            return None
        hidden_out = getattr(base_decoder, "__call__")(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        loss_fn = ChunkedCausalLMLoss(
            chunk_size=self._chunk_cfg["chunk_size"],
            use_checkpoint=self.training,
        )
        return {"loss": loss_fn(hidden_out.last_hidden_state, lm_head, labels)}

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass supporting dict batch unpacking, HF model structure, and mock testing."""
        if isinstance(input_ids, dict):
            labels = input_ids.get("labels", labels)
            attention_mask = input_ids.get("attention_mask", attention_mask)
            input_ids = input_ids.get("input_ids")

        if hasattr(self, "embed") and hasattr(self, "lm_head"):
            # Mock pathway for tests
            h = self.embed(input_ids)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
            return {"loss": loss if loss is not None else torch.tensor(0.0), "logits": logits}

        if self._chunk_cfg["enabled"] and labels is not None:
            chunked_out = self._chunked_forward(
                input_ids, attention_mask, labels, **kwargs
            )
            if chunked_out is not None:
                return chunked_out

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def train_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Unified training step handling dictionary batches seamlessly."""
        if isinstance(batch, dict):
            return self.forward(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        return self.forward(batch)

    def eval_step(self, batch: Any) -> dict[str, Any]:
        """Unified validation step handling dictionary batches seamlessly."""
        if (
            self._chunk_cfg["enabled"]
            and isinstance(batch, dict)
            and batch.get("labels") is not None
            and not (hasattr(self, "embed") and hasattr(self, "lm_head"))
        ):
            base_decoder: Any = getattr(self.model, "model", None) or getattr(
                getattr(self.model, "base_model", None), "model", None
            )
            lm_head = self.model.get_output_embeddings()
            if isinstance(base_decoder, nn.Module) and lm_head is not None:
                hidden_out = getattr(base_decoder, "__call__")(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    return_dict=True,
                )
                loss, pred_ids = chunked_eval_loss_and_preds(
                    hidden_out.last_hidden_state,
                    lm_head,
                    batch.get("labels"),
                    chunk_size=self._chunk_cfg["chunk_size"],
                )
                return {"loss": loss, "preds": pred_ids}

        out = self.train_step(batch)
        return {
            "loss": out["loss"],
            "preds": out.get("logits", out["loss"]),
        }


# Backward-compatible subclass alias
class AIHelperLemGend(GenericCausalLLM):
    """Backward-compatible alias for the official AI Assistant LLM."""
