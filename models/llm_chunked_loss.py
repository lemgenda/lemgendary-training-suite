"""Memory-frugal chunked linear cross-entropy for large-vocabulary causal LLMs."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce_sum(
    lm_head: nn.Module,
    hidden_chunk: torch.Tensor,
    label_chunk: torch.Tensor,
) -> torch.Tensor:
    logits_chunk = lm_head(hidden_chunk).float()
    return F.cross_entropy(logits_chunk, label_chunk, reduction="sum")


def _active_token_pairs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.view(-1)
    valid_mask = flat_labels.ne(-100)
    if int(valid_mask.sum().item()) == 0:
        return None
    return flat_hidden[valid_mask], flat_labels[valid_mask]


class ChunkedCausalLMLoss(nn.Module):
    """Compute exact causal LM cross-entropy without materializing full ``[B, S, V]`` logits."""

    def __init__(self, chunk_size: int = 256, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.chunk_size = max(1, chunk_size)
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked causal LM cross-entropy loss over active (``!= -100``) tokens."""
        active = _active_token_pairs(hidden_states, labels)
        if active is None:
            return hidden_states.sum() * 0.0
        active_hidden, active_labels = active

        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        num_active = active_labels.size(0)
        for start in range(0, num_active, self.chunk_size):
            end = min(start + self.chunk_size, num_active)
            h_chunk = active_hidden[start:end]
            y_chunk = active_labels[start:end]
            if self.use_checkpoint and self.training and h_chunk.requires_grad:
                chunk_loss = checkpoint(
                    _chunk_ce_sum,
                    lm_head,
                    h_chunk,
                    y_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = _chunk_ce_sum(lm_head, h_chunk, y_chunk)
            total_loss = total_loss + chunk_loss

        return total_loss / float(num_active)


def _eval_chunk_step(
    lm_head: nn.Module,
    h_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    logits_chunk = lm_head(h_chunk).float()
    mask = y_chunk.ne(-100)
    n_valid = int(mask.sum().item())
    chunk_loss = (
        F.cross_entropy(logits_chunk[mask], y_chunk[mask], reduction="sum")
        if n_valid > 0
        else torch.zeros((), device=h_chunk.device, dtype=torch.float32)
    )
    return chunk_loss, n_valid, logits_chunk.argmax(dim=-1)


def _shift_eval_inputs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    bsz, seq_len, hidden_dim = hidden_states.shape
    shift_hidden = hidden_states[:, :-1, :].contiguous().view(-1, hidden_dim)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return bsz, seq_len, shift_hidden, shift_labels


def _accumulate_eval_chunks(
    lm_head: nn.Module,
    shift_hidden: torch.Tensor,
    shift_labels: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    total_loss = torch.zeros((), device=shift_hidden.device, dtype=torch.float32)
    valid_count = 0
    pred_chunks: list[torch.Tensor] = []
    step = max(1, chunk_size)
    for start in range(0, shift_labels.size(0), step):
        end = min(start + step, shift_labels.size(0))
        c_loss, n_valid, c_preds = _eval_chunk_step(
            lm_head, shift_hidden[start:end], shift_labels[start:end]
        )
        total_loss = total_loss + c_loss
        valid_count += n_valid
        pred_chunks.append(c_preds)
    return total_loss, valid_count, torch.cat(pred_chunks, dim=0)


@torch.no_grad()
def chunked_eval_loss_and_preds(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact eval cross-entropy loss and compact ``[B, S]`` argmax token IDs."""
    bsz, seq_len, shift_hidden, shift_labels = _shift_eval_inputs(hidden_states, labels)
    total_loss, valid_count, flat_preds = _accumulate_eval_chunks(
        lm_head, shift_hidden, shift_labels, chunk_size
    )
    loss = total_loss / max(1, valid_count)
    shift_preds = flat_preds.view(bsz, seq_len - 1)
    pad_col = torch.zeros((bsz, 1), device=hidden_states.device, dtype=shift_preds.dtype)
    return loss, torch.cat([shift_preds, pad_col], dim=1)
