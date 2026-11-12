"""LLM validation metrics, curriculum batching/fraction config and export helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from onnx import TensorProto, helper

from training.export.llm_exporter import _export_fp16_integrated, _kv_cache_disabled
from training.governance.governor_setup import build_curriculum, configured_data_fraction
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import ValBatch, update_llm_tokens
from training.training.val_summary import summarize


def _llm_batch(preds: torch.Tensor, labels: torch.Tensor | None) -> ValBatch:
    return ValBatch(
        preds=preds,
        targets=None,
        sota_cfg={},
        dataset_type="text_instruction",
        device=torch.device("cpu"),
        governor=None,
        labels=labels,
    )


class _ConfiguredModel(torch.nn.Module):
    """A module carrying a Hugging Face-style ``config.use_cache`` flag."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


class TestLlmTokenAccuracy(unittest.TestCase):
    """Next-token accuracy and the LLM summary keys."""

    def setUp(self) -> None:
        # Vocabulary of 4; logits predict token ids [1, 2, 3, 0] at positions 0..3.
        self.preds = torch.nn.functional.one_hot(torch.tensor([[1, 2, 3, 0]]), 4).float()

    def test_shifted_accuracy_ignores_masked_labels(self) -> None:
        """Positions with label -100 are skipped; others compare shifted argmax."""
        labels = torch.tensor([[9, 1, 2, -100]])
        acc = ValAccumulators()
        update_llm_tokens(acc, _llm_batch(self.preds, labels))
        # Shifted pairs: (pred 1, label 1) ok, (pred 2, label 2) ok, label -100 skipped.
        self.assertEqual(acc.llm.total_llm_correct_tokens, 2)
        self.assertEqual(acc.llm.total_llm_valid_tokens, 2)

    def test_non_llm_batch_is_ignored(self) -> None:
        """A batch without LLM labels (non-LLM model) adds no token totals."""
        acc = ValAccumulators()
        update_llm_tokens(acc, _llm_batch(self.preds, None))
        self.assertEqual(acc.llm.total_llm_valid_tokens, 0)

    def test_summary_reports_eval_loss_accuracy_and_perplexity(self) -> None:
        """LLM summaries carry eval_loss, token_accuracy (%) and perplexity."""
        acc = ValAccumulators(total_loss=2.0)
        acc.llm.total_llm_correct_tokens = 3
        acc.llm.total_llm_valid_tokens = 4
        ctx = SimpleNamespace(model_name="ai_helper_lemgend", models=SimpleNamespace(model=None))
        results = summarize(ctx, acc, step_count=2, elapsed=0.0)
        self.assertEqual(results["eval_loss"], 1.0)
        self.assertEqual(results["val_loss"], 1.0)
        self.assertEqual(results["token_accuracy"], 75.0)
        self.assertAlmostEqual(results["perplexity"], 2.7183, places=4)

    def test_token_totals_survive_progress_snapshot(self) -> None:
        """Token totals round-trip through the intra-epoch progress snapshot."""
        acc = ValAccumulators()
        acc.llm.total_llm_correct_tokens = 5
        acc.llm.total_llm_valid_tokens = 8
        restored = ValAccumulators()
        restored.restore(acc.snapshot())
        self.assertEqual(restored.llm.total_llm_valid_tokens, 8)


class TestCurriculumConfig(unittest.TestCase):
    """Data fraction, batch and accumulation resolved from the model config."""

    def test_llm_curriculum_block(self) -> None:
        """``curriculum.data_fraction`` and ``accumulation_steps`` are honoured."""
        info = {
            "category": "llm",
            "dataset_type": "text_instruction",
            "batch_size": 2,
            "input_size": 2048,
            "curriculum": {"data_fraction": 1.0, "accumulation_steps": 12},
        }
        cur = build_curriculum(info, {}, "text_instruction")
        self.assertEqual(cur.current_fraction, 1.0)
        self.assertEqual(cur.current_batch, 2)
        self.assertEqual(cur.current_acc, 12)

    def test_llm_default_batch_and_derived_accumulation(self) -> None:
        """An LLM without a batch size starts at 2 with acc = target / batch."""
        info = {"category": "llm", "curriculum": {"target_effective_batch": 24}}
        cur = build_curriculum(info, {}, "text_instruction")
        self.assertEqual(cur.current_batch, 2)
        self.assertEqual(cur.current_acc, 12)
        self.assertEqual(cur.current_fraction, 1.0)

    def test_configured_fraction_precedence(self) -> None:
        """``sample_fraction`` wins over ``curriculum.data_fraction``; None when unset."""
        both = {"sample_fraction": 0.5, "curriculum": {"data_fraction": 0.8}}
        self.assertEqual(configured_data_fraction(both), 0.5)
        self.assertEqual(configured_data_fraction({"curriculum": {"data_fraction": 0.8}}), 0.8)
        self.assertIsNone(configured_data_fraction({}))


class TestLlmExportHelpers(unittest.TestCase):
    """KV-cache toggling and the FP16 copy step of the LLM exporter."""

    def test_kv_cache_restored_after_export(self) -> None:
        """``use_cache`` is off inside the context and restored afterwards."""
        model = _ConfiguredModel()
        with _kv_cache_disabled(model):
            self.assertFalse(model.config.use_cache)
        self.assertTrue(model.config.use_cache)

    def test_fp16_copy_requires_fp32_source(self) -> None:
        """Without an FP32 graph the FP16 step reports nothing; with one it converts or copies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fp32, fp16 = tmp / "m_FP32.onnx", tmp / "m.onnx"
            self.assertEqual(_export_fp16_integrated(fp32, fp16), {})
            # Create a valid minimal ONNX model with a float32 tensor
            node = helper.make_node("Relu", ["X"], ["Y"])
            graph = helper.make_graph(
                [node],
                "test",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
            )
            model = helper.make_model(graph, producer_name="test")
            onnx.save(model, str(fp32))
            res = _export_fp16_integrated(fp32, fp16)
            self.assertIn("onnx_fp16", res)
            self.assertEqual(res["onnx_fp16"], fp16)
            self.assertTrue(fp16.exists())


if __name__ == "__main__":
    unittest.main()
