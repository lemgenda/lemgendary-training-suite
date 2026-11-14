"""Unit tests for Kaggle TPU v5e-8 bundle resolution and dual-regime LLM optimizations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.llm_chunked_loss import ChunkedCausalLMLoss, chunked_eval_loss_and_preds
from training.cloud.kernel_bundle import BundleRequest, bundle_spec, kernel_metadata
from training.data.instruction_dataset import collate_instruction_batch
from training.training.val_metrics import ValAccumulators, ValBatch, update_llm_tokens


def test_ai_helper_lemgend_resolves_tpu_v5e8() -> None:
    """Verify ai_helper_lemgend resolves to Kaggle TPU v5e-8."""
    req = BundleRequest(model_key="ai_helper_lemgend", kernel_owner="lemgend")
    spec, _ = bundle_spec(req)
    meta = kernel_metadata(spec)
    assert spec.accelerator.id == "tpu-v5e8"
    assert meta["enable_tpu"] is True
    assert meta["enable_gpu"] is False
    assert meta["machine_shape"] == "TpuV5E8"


def test_chunked_causal_lm_loss_matches_full_ce() -> None:
    """Verify chunked cross-entropy matches full-logit cross-entropy."""
    torch.manual_seed(42)
    bsz, seq_len, hidden_dim, vocab_size = 2, 16, 32, 128
    hidden = torch.randn(bsz, seq_len, hidden_dim, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)

    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    labels = torch.randint(0, vocab_size, (bsz, seq_len))
    labels[:, :3] = -100

    loss_fn = ChunkedCausalLMLoss(chunk_size=5, use_checkpoint=False)
    loss_chunked = loss_fn(hidden, lm_head, labels)
    loss_chunked.backward()

    logits_ref = lm_head(hidden_ref)[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), shift_labels.view(-1))
    loss_ref.backward()

    assert torch.allclose(loss_chunked, loss_ref, rtol=1e-5, atol=1e-5)
    assert hidden.grad is not None and hidden_ref.grad is not None
    assert torch.allclose(hidden.grad, hidden_ref.grad, rtol=1e-4, atol=1e-4)


def test_chunked_eval_and_2d_val_metrics() -> None:
    """Verify chunked eval produces 2D pred_ids compatible with update_llm_tokens."""
    torch.manual_seed(7)
    bsz, seq_len, hidden_dim, vocab_size = 2, 12, 16, 64
    hidden = torch.randn(bsz, seq_len, hidden_dim)
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    labels = torch.randint(0, vocab_size, (bsz, seq_len))
    labels[:, :2] = -100

    loss, pred_ids = chunked_eval_loss_and_preds(hidden, lm_head, labels, chunk_size=4)
    assert pred_ids.shape == (bsz, seq_len)
    assert loss.item() > 0.0

    acc = ValAccumulators()
    batch = ValBatch(
        preds=pred_ids,
        targets=None,
        labels=labels,
        sota_cfg={},
        dataset_type="text_instruction",
        device=torch.device("cpu"),
        governor=None,
    )
    update_llm_tokens(acc, batch)
    assert acc.llm.total_llm_valid_tokens == bsz * (seq_len - 2)


def test_static_shape_collation_for_xla() -> None:
    """Verify static-shape padding produces invariant [B, S] tensors for XLA."""
    items = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7, 8], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 6, 7, 8], dtype=torch.long),
        },
    ]
    out = collate_instruction_batch(
        items, pad_token_id=0, pad_to_max_length=True, max_seq_length=16
    )
    assert out["input_ids"].shape == (2, 16)
    assert out["attention_mask"].shape == (2, 16)
    assert out["labels"].shape == (2, 16)
