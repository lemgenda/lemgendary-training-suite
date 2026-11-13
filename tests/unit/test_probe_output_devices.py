"""The export probe compares PyTorch and ONNX Runtime outputs on the CPU.

Kaggle run 5 of ``upn_v2`` failed both ONNX release exports at the end of epoch 1 with
"Expected all tensors to be on the same device, but found at least two devices, cuda:0 and
cpu": the model's outputs were CUDA tensors and ONNX Runtime's were CPU tensors. The probe
now moves both sides to :data:`PROBE_DEVICE` first. Outputs living on another device are
stood in for by a tensor double that records where it is moved; with a GPU the end-to-end
check runs a CUDA model through ``build_verified_program``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest
import torch
from torch import nn

from training.export.common import default_graph_spec
from training.export.dynamo_onnx import (
    PROBE_DEVICE,
    ProbeTolerance,
    build_verified_program,
    compare_outputs,
)

_VALUES = torch.tensor([[0.25, -1.5], [2.0, 0.0]])


@dataclass
class _OtherDeviceTensor:
    """Output on a device other than the CPU: records the device it is moved to."""

    value: torch.Tensor
    moves: list[torch.device | str | None] = field(default_factory=list)

    def detach(self) -> _OtherDeviceTensor:
        """Detached view (itself)."""
        return self

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
        """Record the target device; the CPU copy is the only one the test hands out."""
        self.moves.append(device)
        return self.value.to(dtype=dtype or self.value.dtype)


def test_both_sides_move_to_the_probe_device() -> None:
    """PyTorch and ONNX Runtime outputs are both moved to the CPU before they are compared."""
    expected = _OtherDeviceTensor(_VALUES.clone())
    actual = _OtherDeviceTensor(_VALUES.clone())
    result = compare_outputs(
        cast(list[torch.Tensor], [expected]), cast(list[torch.Tensor], [actual]), ProbeTolerance()
    )
    assert result.ok
    assert expected.moves == [PROBE_DEVICE]
    assert actual.moves == [PROBE_DEVICE]
    assert PROBE_DEVICE == torch.device("cpu")


def test_differing_outputs_still_fail() -> None:
    """Moving to the CPU does not loosen the comparison."""
    result = compare_outputs([_VALUES], [_VALUES + 1.0], ProbeTolerance())
    assert not result.ok
    assert result.max_abs_diff == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_model_exports_with_a_verified_probe() -> None:
    """A model on CUDA is exported and verified against ONNX Runtime on the CPU."""
    model = nn.Conv2d(3, 4, 3, padding=1).cuda().eval()
    sample = torch.randn(2, 3, 16, 16, device="cuda")
    verified = build_verified_program(model, (sample,), default_graph_spec(sample))
    assert all(probe.ok for probe in verified.probes)
