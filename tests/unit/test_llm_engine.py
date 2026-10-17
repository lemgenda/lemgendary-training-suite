"""Unit tests for AIHelperLemGend LLM fine-tuning engine and instruction dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from models.ai_helper_llm import AIHelperLemGend
from training.data.instruction_dataset import (
    InstructionDataset,
    collate_instruction_batch,
)


class DummyMockTokenizer:
    """Deterministic mock tokenizer for testing instruction prompt masking."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Simple ASCII-based deterministic token mapping
        tokens = [max(1, ord(c) % 500) for c in text.strip()]
        return tokens or [1]


class TestLLMEngine(unittest.TestCase):
    """Verifies AIHelperLemGend model wrapper and instruction dataset processing."""

    def test_ai_helper_model_instantiation(self) -> None:
        """Ensure AIHelperLemGend model initializes cleanly with mock or HF backbone."""
        model = AIHelperLemGend(base_model_name_or_path="mock", use_lora=False)
        self.assertIsNotNone(model)

        # Verify forward pass
        dummy_input = torch.randint(1, 1000, (2, 16), dtype=torch.long)
        dummy_labels = torch.randint(1, 1000, (2, 16), dtype=torch.long)
        outputs = model(input_ids=dummy_input, labels=dummy_labels)

        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)
        self.assertIsNotNone(outputs["loss"])
        self.assertFalse(torch.isnan(outputs["loss"]))

    def test_instruction_dataset_parsing_and_masking(self) -> None:
        """Verify multi-turn JSONL parsing and assistant-only supervised label masking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_file = Path(tmp_dir) / "instructions.jsonl"
            sample_data = {
                "messages": [
                    {"role": "system", "content": "You are LemGendary AI Assistant."},
                    {"role": "user", "content": "Optimize my restoration pipeline."},
                    {"role": "assistant", "content": "Here is the recommended configuration."},
                ]
            }
            with open(jsonl_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(sample_data) + "\n")

            tokenizer = DummyMockTokenizer()
            dataset = InstructionDataset(jsonl_path=jsonl_file, tokenizer=tokenizer, max_length=128)

            self.assertEqual(len(dataset), 1)
            item = dataset[0]

            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
            self.assertIn("labels", item)

            input_ids = item["input_ids"]
            labels = item["labels"]
            self.assertEqual(input_ids.shape, labels.shape)

            # Assert prompt masking: system + user tokens must be masked with -100
            masked_count = (labels == -100).sum().item()
            unmasked_count = (labels != -100).sum().item()

            self.assertGreater(masked_count, 0, "System and user tokens must be masked with -100")
            self.assertGreater(unmasked_count, 0, "Assistant response tokens must have valid supervised labels")

    def test_collate_instruction_batch(self) -> None:
        """Verify dynamic batch padding and attention mask alignment."""
        sample_a = {
            "input_ids": torch.tensor([10, 20, 30], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 30], dtype=torch.long),
        }
        sample_b = {
            "input_ids": torch.tensor([40, 50, 60, 70, 80], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 60, 70, 80], dtype=torch.long),
        }

        batch = collate_instruction_batch([sample_a, sample_b], pad_token_id=0)

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["attention_mask"].shape, (2, 5))
        self.assertEqual(batch["labels"].shape, (2, 5))

        # Sample A must be padded from len 3 to len 5 with pad_token_id=0 and label=-100
        self.assertEqual(batch["input_ids"][0, 3].item(), 0)
        self.assertEqual(batch["input_ids"][0, 4].item(), 0)
        self.assertEqual(batch["attention_mask"][0, 3].item(), 0)
        self.assertEqual(batch["attention_mask"][0, 4].item(), 0)
        self.assertEqual(batch["labels"][0, 3].item(), -100)
        self.assertEqual(batch["labels"][0, 4].item(), -100)


if __name__ == "__main__":
    unittest.main()
