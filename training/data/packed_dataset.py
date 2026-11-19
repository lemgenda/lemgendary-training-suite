"""Sequence packing and block-diagonal attention masking for LLM instruction datasets.

Packs multiple variable-length conversation sequences into fixed-length windows,
eliminating padding waste while generating block-diagonal 2D attention masks and
resetting position IDs to prevent cross-document attention contamination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class PackedSequence:
    """A single packed token sequence composed of multiple independent segments."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    segment_lengths: list[int]


def build_block_diagonal_mask(segment_lengths: list[int], total_length: int) -> torch.Tensor:
    """Build a 2D causal attention mask preventing cross-sequence attention.

    Each segment can attend strictly to preceding tokens within its own segment,
    forming block-diagonal lower-triangular attention blocks.

    Args:
        segment_lengths: List of token counts for each packed sequence.
        total_length: Total context window length.

    Returns:
        Tensor of shape [total_length, total_length] with 1 for allowed attention,
        0 for masked attention.
    """
    mask = torch.zeros((total_length, total_length), dtype=torch.long)
    curr = 0
    for length in segment_lengths:
        if length <= 0:
            continue
        end = min(curr + length, total_length)
        seg_len = end - curr
        # Lower-triangular block for this segment
        mask[curr:end, curr:end] = torch.tril(torch.ones((seg_len, seg_len), dtype=torch.long))
        curr = end
        if curr >= total_length:
            break
    return mask


@dataclass
class _PackBuffer:
    """Working accumulator for packing multiple segments into one window."""

    ids: list[torch.Tensor]
    labels: list[torch.Tensor]
    positions: list[torch.Tensor]
    lengths: list[int]
    total_len: int = 0

    def clear(self) -> None:
        """Reset the working accumulator buffers."""
        self.ids.clear()
        self.labels.clear()
        self.positions.clear()
        self.lengths.clear()
        self.total_len = 0


def _flush_buffer(
    buf: _PackBuffer,
    max_seq_length: int,
    pad_token_id: int,
) -> PackedSequence:
    slack = max_seq_length - buf.total_len
    if slack > 0:
        buf.ids.append(torch.full((slack,), pad_token_id, dtype=torch.long))
        buf.labels.append(torch.full((slack,), IGNORE_INDEX, dtype=torch.long))
        buf.positions.append(torch.zeros(slack, dtype=torch.long))
        buf.lengths.append(slack)

    packed_ids = torch.cat(buf.ids, dim=0)
    packed_labels = torch.cat(buf.labels, dim=0)
    packed_positions = torch.cat(buf.positions, dim=0)
    packed_mask = build_block_diagonal_mask(buf.lengths, max_seq_length)

    return PackedSequence(
        input_ids=packed_ids,
        attention_mask=packed_mask,
        labels=packed_labels,
        position_ids=packed_positions,
        segment_lengths=list(buf.lengths),
    )


def pack_instruction_samples(
    samples: list[dict[str, torch.Tensor]],
    max_seq_length: int = 512,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """Pack variable-length tokenized instruction samples into continuous windows.

    Args:
        samples: List of dictionaries containing 'input_ids' and 'labels'.
        max_seq_length: Target packed sequence window length.
        pad_token_id: Token ID used to pad any remaining slack in the final window.

    Returns:
        List of PackedSequence objects.
    """
    if not samples:
        return []

    packed_records: list[PackedSequence] = []
    buf = _PackBuffer(ids=[], labels=[], positions=[], lengths=[])

    for sample in samples:
        ids = sample["input_ids"].reshape(-1)
        labels = sample.get("labels", ids).reshape(-1)
        seq_len = min(int(ids.size(0)), max_seq_length)
        ids = ids[:seq_len]
        labels = labels[:seq_len]

        if buf.total_len + seq_len > max_seq_length and buf.ids:
            packed_records.append(_flush_buffer(buf, max_seq_length, pad_token_id))
            buf.clear()

        buf.ids.append(ids)
        buf.labels.append(labels)
        buf.positions.append(torch.arange(seq_len, dtype=torch.long))
        buf.lengths.append(seq_len)
        buf.total_len += seq_len

    if buf.ids:
        packed_records.append(_flush_buffer(buf, max_seq_length, pad_token_id))

    return packed_records
