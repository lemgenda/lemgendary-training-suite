"""LemGendary AI Studio Assistant Causal Language Model.

Wraps HuggingFace AutoModelForCausalLM with optional LoRA / PEFT adapters,
quantization shielding, and provides standardized forward pass for instruction tuning.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("lemtrain.models.ai_helper")


class AIHelperLemGend(nn.Module):
    """Causal Language Model fine-tuning wrapper for LemGendary AI Assistant."""

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        checkpoint: str | None = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_model_name = base_model_name_or_path
        self.use_lora = use_lora
        self.model: Any = None
        self.tokenizer: Any = None

        logger.info("Initializing AIHelperLemGend backbone: %s", self.base_model_name)

        if self.base_model_name == "mock" or kwargs.get("mock_mode", False):
            self._init_mock_model()
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side="right",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = None
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed; loading without 4-bit quantization.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quant_config,
                torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            if self.use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

                    if load_in_4bit:
                        self.model = prepare_model_for_kbit_training(self.model)

                    target_modules = kwargs.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                    peft_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=target_modules,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    self.model = get_peft_model(self.model, peft_config)
                    logger.info("Injected LoRA adapter into AIHelperLemGend backbone (r=%d)", lora_r)
                except ImportError:
                    logger.warning("PEFT not installed. AIHelperLemGend operating in standard mode.")

        except Exception as exc:
            logger.warning(
                "Could not initialize full transformers model (%s). Using lightweight mock for testing.", exc
            )
            self._init_mock_model()

    def _init_mock_model(self) -> None:
        """Lightweight mock embedding and linear projection for testing and verification."""
        vocab_size = 32000
        hidden_dim = 256
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass supporting both HF model structure and mock testing."""
        if hasattr(self, "embed") and hasattr(self, "lm_head"):
            # Mock pathway for tests
            h = self.embed(input_ids)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return {"loss": loss if loss is not None else torch.tensor(0.0), "logits": logits}

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
