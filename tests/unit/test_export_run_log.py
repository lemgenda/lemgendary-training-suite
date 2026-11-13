"""A successful export writes its parity result and release files to the run log.

Kaggle run 6 of ``upn_v2`` exported both ONNX variants, but the log showed only
"[EXPORT] Initiating ...": the trainer configures no INFO handler, so the passed probe and
the written files were reported through ``logging`` alone and never appeared.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from training.export.common import OnnxGraphSpec, default_graph_spec
from training.export.dynamo_onnx import (
    ProbeResult,
    ProbeTolerance,
    build_verified_program,
    report_parity,
)
from training.export.universal_exporter import export_summary


def test_passed_probe_prints_its_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    """The parity line names the probe shape, the worst difference and the tolerance."""
    line = report_parity(ProbeResult(ok=True, max_abs_diff=2.5e-6), ProbeTolerance(), "1x3x48x48")
    assert line == (
        "[EXPORT] ONNX FP32 parity probe at 1x3x48x48: passed, max abs diff 2.5e-06 "
        "(atol 0.001, rtol 0.01)"
    )
    assert capsys.readouterr().out.strip() == line


def test_failed_probe_names_its_reason() -> None:
    """A failed probe keeps its detail in the line."""
    line = report_parity(
        ProbeResult(ok=False, max_abs_diff=0.5, detail="outputs differ"), ProbeTolerance(), "2x3"
    )
    assert "failed (outputs differ), max abs diff 0.5" in line


def test_static_graph_says_no_probe_ran() -> None:
    """Without dynamic axes the line says the probe was not run instead of staying silent."""
    assert "not run" in report_parity(None, ProbeTolerance(), "")


def test_verified_export_prints_the_parity_line(capsys: pytest.CaptureFixture[str]) -> None:
    """``build_verified_program`` prints the passed probe at the grown shape."""
    sample = torch.randn(2, 3, 16, 16)
    build_verified_program(
        nn.Conv2d(3, 4, 3, padding=1).eval(), (sample,), default_graph_spec(sample)
    )
    out = capsys.readouterr().out
    assert "[EXPORT] ONNX FP32 parity probe at 3x3x24x24: passed, max abs diff" in out


def test_static_export_prints_that_no_probe_ran(capsys: pytest.CaptureFixture[str]) -> None:
    """A graph without dynamic axes reports that it skipped the probe."""
    sample = torch.randn(1, 3, 8, 8)
    graph = OnnxGraphSpec(input_names=["input"], output_names=["output"])
    build_verified_program(nn.Conv2d(3, 2, 1).eval(), (sample,), graph)
    assert "parity probe not run" in capsys.readouterr().out


def test_export_summary_lists_the_written_files() -> None:
    """The summary names every release file once, sorted by key."""
    line = export_summary(
        "upn_v2",
        {"onnx_fp32": Path("a") / "M_FP32.onnx", "pytorch": Path("b") / "M.pt"},
    )
    assert line == "[EXPORT] upn_v2: 2 release file(s) written: onnx_fp32=M_FP32.onnx, pytorch=M.pt"


def test_export_summary_flags_an_empty_export() -> None:
    """An export that wrote nothing says so."""
    assert export_summary("upn_v2", {}) == "[EXPORT] upn_v2: no release files were written."
