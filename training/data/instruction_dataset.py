"""Instruction Dataset Loader for LemGendizedAIHelperLemGend.

Parses conversation JSONL files, formats system/user/assistant message turns,
and constructs tokenized input_ids, attention_mask, and labels with proper
prompt masking (-100 on system and user tokens) so loss is computed exclusively
on assistant outputs.
"""

from __future__ import annotations

import collections.abc
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("lemtrain.data.instruction")

try:
    import jinja2

    _JINJA_TEMPLATE_ERRORS = (jinja2.TemplateError,)
except ImportError as _jinja_err:
    logger.debug(
        "jinja2 unavailable; chat template errors limited to builtin types: %s", _jinja_err
    )
    _JINJA_TEMPLATE_ERRORS = ()

# Failure modes of tokenizer.apply_chat_template (missing/invalid template, bad message shape).
_CHAT_TEMPLATE_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    RuntimeError,
    *_JINJA_TEMPLATE_ERRORS,
)


IGNORE_INDEX = -100
"""Label value cross-entropy skips (system and user prompt tokens)."""


def _template_ids(res: Any) -> torch.Tensor:
    """Normalize an ``apply_chat_template`` result to a 1-D ``torch.long`` id tensor."""
    if hasattr(res, "input_ids"):
        raw_ids = res.input_ids
    elif isinstance(res, collections.abc.Mapping) and "input_ids" in res:
        raw_ids = res["input_ids"]
    elif isinstance(res, (list, tuple)) and res and isinstance(res[0], (list, tuple)):
        raw_ids = res[0]
    else:
        raw_ids = res

    encoding_ids = getattr(raw_ids, "ids", None)
    if encoding_ids is not None:
        input_ids = torch.tensor(encoding_ids, dtype=torch.long)
    elif isinstance(raw_ids, torch.Tensor):
        input_ids = raw_ids.reshape(-1).to(dtype=torch.long)
    else:
        input_ids = torch.tensor(raw_ids, dtype=torch.long).reshape(-1)
    return input_ids


class InstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for causal language model instruction fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any | None = None,
        max_length: int = 2048,
        sample_fraction: float = 1.0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_fraction = float(sample_fraction)
        self.all_samples: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL instruction dataset not found at {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        self.all_samples.append(json.loads(line_str))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping corrupted JSON line: %s", exc)

        self._apply_sample_fraction()

        logger.info(
            "Loaded %d/%d instruction samples from %s (fraction=%.2f, max_length=%d)",
            len(self.samples),
            len(self.all_samples),
            self.jsonl_path.name,
            self.sample_fraction,
            self.max_length,
        )

    def _apply_sample_fraction(self) -> None:
        """Apply deterministic curriculum sample-fraction slicing to ``self.all_samples``."""
        self.samples = list(self.all_samples)
        if self.sample_fraction < 1.0 and self.samples:
            rng = random.Random(42)
            rng.shuffle(self.samples)
            target_count = max(1, int(len(self.samples) * self.sample_fraction))
            self.samples = self.samples[:target_count]

    def update_strategy(
        self,
        size: Any = None,
        fraction: float | None = None,
        stress: float | None = None,
    ) -> None:
        """Dynamic curriculum scaling for instruction datasets."""
        _ = (size, stress)
        if fraction is not None:
            self.sample_fraction = float(fraction)
            self._apply_sample_fraction()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        messages = sample.get("messages", [])

        if self.tokenizer is None:
            # Fallback when tokenizer is absent (unit tests / mock mode)
            dummy_len = min(64, self.max_length)
            return {
                "input_ids": torch.randint(1, 1000, (dummy_len,), dtype=torch.long),
                "attention_mask": torch.ones(dummy_len, dtype=torch.long),
                "labels": torch.randint(1, 1000, (dummy_len,), dtype=torch.long),
            }

        # Apply chat template or manual turn construction
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self._chat_template_encode(self.tokenizer, messages)
            if encoded is not None:
                return encoded
        return self._manual_encode(self.tokenizer, messages)

    def _chat_template_encode(
        self, tokenizer: Any, messages: list
    ) -> dict[str, torch.Tensor] | None:
        """Tokenize with the chat template, prompt tokens masked; ``None`` when it fails.

        The prompt is every message before the final assistant turn, rendered with
        ``add_generation_prompt=True`` so the assistant header counts as prompt. Its token
        length is masked to ``IGNORE_INDEX`` exactly as :meth:`_manual_encode` does.
        """
        try:
            input_ids = _template_ids(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            )
            prompt_len = self._template_prompt_len(tokenizer, messages)
        except _CHAT_TEMPLATE_ERRORS as exc:
            logger.debug("apply_chat_template failed: %s. Using manual tokenization.", exc)
            return None

        labels = input_ids.clone()
        labels[: min(prompt_len, labels.numel())] = IGNORE_INDEX
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }

    def _template_prompt_len(self, tokenizer: Any, messages: list) -> int:
        """Token count of the chat-template prompt (all turns before the last assistant)."""
        if not messages or messages[-1].get("role") != "assistant":
            return 0
        prompt_ids = _template_ids(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        )
        return prompt_ids.numel()

    def _manual_encode(self, tokenizer: Any, messages: list) -> dict[str, torch.Tensor]:
        """ChatML turns with prompt tokens masked out of the labels."""
        # Manual turn concatenation with label masking
        formatted_prompt = ""
        assistant_reply = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("system", "user"):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif role == "assistant":
                assistant_reply += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        reply_tokens = tokenizer.encode(assistant_reply, add_special_tokens=False)

        combined_tokens = (prompt_tokens + reply_tokens)[: self.max_length]
        prompt_len = min(len(prompt_tokens), len(combined_tokens))

        input_ids = torch.tensor(combined_tokens, dtype=torch.long)
        attention_mask = torch.ones(len(combined_tokens), dtype=torch.long)

        # Label masking: set -100 on prompt tokens so loss is only calculated on assistant tokens
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_instruction_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    pad_to_max_length: bool = False,
    max_seq_length: int = 512,
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad input_ids, attention_mask, and labels to static or aligned sequence length."""
    if pad_to_max_length:
        max_len = max_seq_length
    else:
        max_len = max(item["input_ids"].size(0) for item in batch)
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            rem = max_len % pad_to_multiple_of
            if rem != 0:
                max_len += pad_to_multiple_of - rem

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        cur_len = item["input_ids"].size(0)
        if cur_len > max_len:
            padded_ids = item["input_ids"][:max_len]
            padded_mask = item["attention_mask"][:max_len]
            padded_labels = item["labels"][:max_len]
        elif cur_len < max_len:
            pad_len = max_len - cur_len
            padded_ids = torch.cat(
                [item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
            padded_mask = torch.cat(
                [item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
            padded_labels = torch.cat(
                [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )
        else:
            padded_ids = item["input_ids"]
            padded_mask = item["attention_mask"]
            padded_labels = item["labels"]

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)

    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_attention_mask, dim=0),
        "labels": torch.stack(batch_labels, dim=0),
    }
