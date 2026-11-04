"""Tensor placement helpers shared by the training loops and parallel strategies."""

from __future__ import annotations

from typing import Any

import torch


def to_device(value: Any, device: torch.device) -> Any:
    """Move a tensor, or the tensors of a dict, to ``device`` (non-blocking).

    Other values (and non-tensor dict entries) are returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in value.items()
        }
    return value
