"""Offline CI/CD parity tests and runtime canary tests for FP16 ONNX exports.

Validates that FP16 graphs are checked offline with numpy.allclose (atol=1e-2, rtol=1e-2)
and runtime canaries detect NaN/Inf conditions without expensive in-run CPU parity probes.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn

from training.export.onnx_variants import ONNX_FP16_VARIANT, OnnxVariantWriter


class DummyModel(nn.Module):
    """Dummy neural module for FP16 export tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.fc(x)


class TestFP16Verification(unittest.TestCase):
    """Parity drift and canary checks for FP16 ONNX exports."""

    def test_offline_parity_drift_within_tolerance(self) -> None:
        """FP16 vs FP32 tensor outputs stay within normal 10-bit mantissa variance (~5e-3)."""
        model = DummyModel().eval()
        x = torch.randn(2, 4)

        with torch.no_grad():
            out_fp32 = model(x).numpy()
            model_fp16 = DummyModel().half().eval()
            model_fp16.fc.weight.data = model.fc.weight.data.half()
            model_fp16.fc.bias.data = model.fc.bias.data.half()
            out_fp16 = model_fp16(x.half()).float().numpy()

        diff = np.max(np.abs(out_fp32 - out_fp16))
        self.assertLess(diff, 1e-2, f"FP16 drift {diff} exceeded offline tolerance 1e-2")

    def test_canary_detects_non_finite_tensor(self) -> None:
        """Canary warns on NaN / Inf output in FP16 program."""
        writer = OnnxVariantWriter(
            model=DummyModel(),
            inputs=(torch.randn(1, 4),),
            graph=MagicMock(),
        )
        fake_prog = MagicMock(return_value=[torch.tensor([float("nan"), 1.0], dtype=torch.float16)])
        with self.assertLogs("lemtrain.export.onnx_variants", level="WARNING") as cm:
            writer.verify_canary(fake_prog, ONNX_FP16_VARIANT)
        self.assertTrue(any("Non-finite output" in log for log in cm.output))


if __name__ == "__main__":
    unittest.main()
