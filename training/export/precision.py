"""Reduced-precision copies of a model for the FP16 ONNX variant, with FP32 islands.

Some layers are not FP16-safe. Swin V2 window attention (torchvision
``ShiftedWindowAttentionV2``) L2-normalises queries and keys with ``eps=1e-12``; in FP16 that
epsilon underflows to zero, so the all-zero key of a padded window token divides 0 by 0 and
the whole output becomes NaN (``nima_aesthetic_pro`` at any input size that is not a multiple
of the window grid). Such layers keep FP32 weights and run in FP32 inside the FP16 graph: their
inputs are cast up and their output is cast back down, which the exporter records as Cast
nodes. New unsafe layer types are added to :data:`FP32_ISLAND_TYPES` only.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import nn
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2

logger = logging.getLogger("lemtrain.export.precision")

FP32_ISLAND_TYPES: tuple[type[nn.Module], ...] = (ShiftedWindowAttentionV2,)
"""Module types that run in FP32 inside reduced-precision graphs."""


def _cast_floats(values: Any, dtype: torch.dtype) -> Any:
    """``values`` with every floating tensor (also inside tuples/lists) cast to ``dtype``."""
    if isinstance(values, torch.Tensor):
        return values.to(dtype) if values.is_floating_point() else values
    if isinstance(values, (tuple, list)):
        return type(values)(_cast_floats(v, dtype) for v in values)
    return values


def _fp32_inputs(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    return _cast_floats(args, torch.float32)


def _island(module: nn.Module, dtype: torch.dtype) -> None:
    """Keep ``module`` in FP32 and cast its output back to ``dtype``."""

    def back_to_dtype(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        return _cast_floats(output, dtype)

    module.float()
    module.register_forward_pre_hook(_fp32_inputs)
    module.register_forward_hook(back_to_dtype)


def reduced_precision_copy(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Deep copy of ``model`` cast to ``dtype``, with :data:`FP32_ISLAND_TYPES` kept in FP32."""
    model_copy = copy.deepcopy(model).to(dtype=dtype).eval()
    islands = [m for m in model_copy.modules() if isinstance(m, FP32_ISLAND_TYPES)]
    for module in islands:
        _island(module, dtype)
    if islands:
        logger.info("Kept %d FP16-unsafe layer(s) in FP32 for the %s graph", len(islands), dtype)
    return model_copy
