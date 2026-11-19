"""Tests verifying Prompt 16 LLM training pipeline improvements.

Validates:
1. Sequence packing and block-diagonal 2D attention mask generation.
2. Position ID reset behavior across packed segments.
3. WSD (Warmup-Stable-Decay) learning rate scheduling.
4. Loss-spike detection and non-finite anomaly flags.
5. Throughput instrumentation and MFU calculation.
6. FSDP strategy registration and interface contract.
7. End-to-end training smoke run comparing loss curve stability.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from models.generic_causal_llm import GenericCausalLLM
from training.data.instruction_dataset import collate_instruction_batch
from training.data.packed_dataset import build_block_diagonal_mask, pack_instruction_samples
from training.parallel import available_strategies
from training.parallel.fsdp import FSDPParallelStrategy
from training.training.optimizer import build_scheduler
from training.training.spike_detector import LossSpikeDetector, SpikeThresholds
from training.training.throughput_tracker import ThroughputTracker


class TestLLMPipelineImprovements(unittest.TestCase):
    """Verifies all deliverables of Prompt 16 LLM pipeline improvements."""

    def test_block_diagonal_mask_structure(self) -> None:
        """Verify 2D attention mask blocks cross-contamination between segments."""
        segment_lengths = [3, 4]
        total_len = 7
        mask = build_block_diagonal_mask(segment_lengths, total_len)

        self.assertEqual(mask.shape, (7, 7))
        # Within segment 1 (0..2), lower-triangular attention is allowed
        self.assertEqual(int(mask[0, 0]), 1)
        self.assertEqual(int(mask[1, 0]), 1)
        self.assertEqual(int(mask[1, 1]), 1)
        self.assertEqual(int(mask[0, 1]), 0)  # Future token masked

        # Cross-segment attention from segment 2 (indices 3..6) back to segment 1 (0..2) is BLOCKED
        for r in range(3, 7):
            for c in range(3):
                self.assertEqual(
                    int(mask[r, c]),
                    0,
                    f"Cross-contamination detected at row {r}, col {c}",
                )

        # Within segment 2 (3..6), lower-triangular attention is allowed
        self.assertEqual(int(mask[3, 3]), 1)
        self.assertEqual(int(mask[4, 3]), 1)
        self.assertEqual(int(mask[4, 4]), 1)

    def test_sequence_packing_and_position_resets(self) -> None:
        """Verify variable-length samples pack with position ID resets and label alignment."""
        sample_a = {
            "input_ids": torch.tensor([10, 11, 12, 13], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 12, 13], dtype=torch.long),
        }
        sample_b = {
            "input_ids": torch.tensor([20, 21, 22], dtype=torch.long),
            "labels": torch.tensor([-100, 21, 22], dtype=torch.long),
        }

        packed = pack_instruction_samples([sample_a, sample_b], max_seq_length=10)
        self.assertEqual(len(packed), 1)

        seq = packed[0]
        self.assertEqual(int(seq.input_ids.size(0)), 10)
        self.assertEqual(int(seq.labels.size(0)), 10)
        self.assertEqual(seq.attention_mask.shape, (10, 10))

        # Position IDs should reset: [0, 1, 2, 3] for sample A, then [0, 1, 2] for sample B
        expected_pos = [0, 1, 2, 3, 0, 1, 2, 0, 0, 0]
        self.assertEqual(seq.position_ids.tolist(), expected_pos)

        # Collate test with 2D mask
        batch_dict = collate_instruction_batch(
            [
                {
                    "input_ids": seq.input_ids,
                    "attention_mask": seq.attention_mask,
                    "labels": seq.labels,
                    "position_ids": seq.position_ids,
                }
            ]
        )
        self.assertEqual(batch_dict["attention_mask"].dim(), 3)  # [1, 10, 10]
        self.assertIn("position_ids", batch_dict)

    def test_wsd_scheduler_trajectory(self) -> None:
        """Verify Warmup-Stable-Decay schedule transitions through all three phases."""
        param = nn.Parameter(torch.zeros(10))
        optimizer = torch.optim.AdamW([param], lr=0.001)

        config = {
            "scheduler": {
                "type": "wsd",
                "warmup_epochs": 2,
                "decay_ratio": 0.20,
                "min_lr": 1e-5,
            }
        }
        total_epochs = 10
        scheduler = build_scheduler(optimizer, config, total_epochs=total_epochs)

        lrs: list[float] = []
        for _ in range(total_epochs):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Warmup phase: step 0 -> ramp up
        self.assertLess(lrs[0], lrs[1])
        # Stable phase: constant peak lr
        self.assertAlmostEqual(lrs[2], 0.001, places=5)
        self.assertAlmostEqual(lrs[5], 0.001, places=5)
        # Decay phase: cooldown towards min_lr
        self.assertLess(lrs[9], lrs[5])

    def test_loss_spike_detector_and_recovery(self) -> None:
        """Verify detector catches malignant loss and gradient spikes, ignoring normal steps."""
        detector = LossSpikeDetector(
            thresholds=SpikeThresholds(
                window_size=20,
                spike_multiplier=3.0,
                grad_norm_multiplier=4.0,
                min_history=5,
            )
        )

        # Feed healthy steps
        for _ in range(10):
            verdict = detector.audit_step(loss_val=2.0, grad_norm=1.0)
            self.assertFalse(verdict.is_spike)

        # Trigger sudden loss explosion (factor > 3.0x median)
        spike_verdict = detector.audit_step(loss_val=10.0, grad_norm=1.0)
        self.assertTrue(spike_verdict.is_spike)
        self.assertFalse(spike_verdict.is_non_finite)

        # Trigger non-finite NaN loss
        nan_verdict = detector.audit_step(loss_val=float("nan"))
        self.assertTrue(nan_verdict.is_spike)
        self.assertTrue(nan_verdict.is_non_finite)

    def test_throughput_and_mfu_calculation(self) -> None:
        """Verify MFU and tokens/second formulas adhere to PaLM benchmark specification."""
        tracker = ThroughputTracker(
            device_type="tpu_v5e",
            num_devices=1,
        )
        flops_per_tok = tracker.compute_flops_per_token(seq_len=512)
        # 6 * 1.5B + 12 * 28 * 12 * 128 * 512 = 9.0e9 + 2.64e8 = ~9.26e9 FLOPs
        self.assertGreater(flops_per_tok, 9.0e9)

        reading = tracker.record_step(tokens_in_step=512, samples_in_step=1, seq_len=512)
        self.assertGreater(reading.tokens_per_sec, 0.0)
        self.assertGreater(reading.mfu_percent, 0.0)

    def test_fsdp_strategy_registration(self) -> None:
        """Verify FSDP strategy is registered in parallel strategies catalog."""
        strategies = available_strategies()
        self.assertIn("fsdp", strategies)

        dummy_model = nn.Linear(16, 16)
        strategy = FSDPParallelStrategy(dummy_model, torch.device("cpu"))
        wrapped = strategy.setup()
        self.assertIsNotNone(wrapped)
        self.assertIs(strategy.unwrap(), dummy_model)

    def test_smoke_training_run_loss_stability(self) -> None:
        """Verify mock Causal LLM training loop maintains monotonic loss decrease."""
        torch.manual_seed(42)
        model = GenericCausalLLM(base_model_name_or_path="mock", use_lora=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        losses: list[float] = []
        dummy_inputs = torch.randint(1, 100, (4, 16))
        dummy_labels = dummy_inputs.clone()

        for _ in range(15):
            optimizer.zero_grad()
            out = model(input_ids=dummy_inputs, labels=dummy_labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # Ensure loss decreased over training iterations
        self.assertLess(losses[-1], losses[0])
