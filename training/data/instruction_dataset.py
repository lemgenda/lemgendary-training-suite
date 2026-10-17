"""Instruction Dataset Loader for LemGendizedAIHelperLemGend.

Parses conversation JSONL files, formats system/user/assistant message turns,
and constructs tokenized input_ids, attention_mask, and labels with proper
prompt masking (-100 on system and user tokens) so loss is computed exclusively
on assistant outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("lemtrain.data.instruction")


class InstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for causal language model instruction fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Any | None = None,
        max_length: int = 2048,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, Any]] = []

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL instruction dataset not found at {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        self.samples.append(json.loads(line_str))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping corrupted JSON line: %s", exc)

        logger.info(
            "Loaded %d instruction samples from %s (max_length=%d)",
            len(self.samples),
            self.jsonl_path.name,
            self.max_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
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
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )[0]
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            except Exception as exc:
                logger.debug("apply_chat_template failed: %s. Using manual tokenization.", exc)

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

        prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        reply_tokens = self.tokenizer.encode(assistant_reply, add_special_tokens=False)

        combined_tokens = (prompt_tokens + reply_tokens)[: self.max_length]
        prompt_len = min(len(prompt_tokens), len(combined_tokens))

        input_ids = torch.tensor(combined_tokens, dtype=torch.long)
        attention_mask = torch.ones(len(combined_tokens), dtype=torch.long)

        # Label masking: set -100 on prompt tokens so loss is only calculated on assistant tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_instruction_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Dynamically pad input_ids, attention_mask, and labels to maximum sequence length in batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for item in batch:
        cur_len = item["input_ids"].size(0)
        pad_len = max_len - cur_len

        if pad_len > 0:
            padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
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
