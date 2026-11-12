"""Device-memory adapter for every GPU backend and the refits that use it (R2-T4).

The resolution-jump refit, the resume clamp and the OOM relief read device memory through
:mod:`training.hardware.device_memory`. Every backend is mocked here (the test host has
no GPU): the adapters' budgets and estimates, the vision probe on a non-CUDA backend and
the refit/clamp keeping the effective batch exact.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tests.fixtures.training_context import fresh_governance
from training import context_resume
from training.context_data import RunSpec
from training.context_resume import TrainState, _clamp_resumed_batch
from training.governance.curriculum import CurriculumState
from training.hardware import device_memory, probe
from training.hardware.device_memory import (
    DIRECTML_ASSUMED_FREE_GB,
    CpuMemory,
    CudaMemory,
    DeviceMemory,
    DirectMLMemory,
    MemoryBudget,
    MpsMemory,
    XpuMemory,
    device_memory_for,
    is_out_of_memory,
    release_device_memory,
)
from training.hardware.probe import VramProbe, audit_hardware_vram
from training.training import batch_budget

GIB = 1024**3


class FakeMemory(DeviceMemory):
    """A mocked GPU backend: fixed budget, a fixed per-probe peak, call log."""

    def __init__(self, name: str, budget: MemoryBudget | None, peak: int) -> None:
        self.name = name
        self._budget = budget
        self._peak = peak
        self.calls: list[str] = []

    def budget(self) -> MemoryBudget | None:
        self.calls.append("budget")
        return self._budget

    def device_name(self) -> str:
        return f"Fake {self.name}"

    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def release_cache(self) -> None:
        self.calls.append("release_cache")

    def reset_peak(self) -> None:
        self.calls.append("reset_peak")

    def observe(self) -> None:
        self.calls.append("observe")

    def allocated(self) -> int:
        return 0

    def peak_allocated(self) -> int:
        return self._peak


class _Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x``."""
        return self.conv(x)


# --------------------------------------------------------------------- adapter selection


def test_adapter_per_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA/ROCm, XPU, MPS and DirectML get their adapter; the CPU has no budget."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert isinstance(device_memory_for(torch.device("cuda", 1)), CudaMemory)
    assert isinstance(device_memory_for(torch.device("xpu")), XpuMemory)
    assert isinstance(device_memory_for(torch.device("mps")), MpsMemory)
    assert isinstance(device_memory_for(torch.device("privateuseone")), DirectMLMemory)
    cpu = device_memory_for(torch.device("cpu"))
    assert isinstance(cpu, CpuMemory)
    assert cpu.budget() is None


def test_rocm_builds_report_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """HIP builds share the CUDA API; the adapter names the backend ROCm."""
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    assert CudaMemory().name == "rocm"


# --------------------------------------------------------------------- budgets


def test_cuda_budget_counts_unused_reserved_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Free memory is the driver's free plus the allocator's reusable cache."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (4 * GIB, 16 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 3 * GIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 1 * GIB)
    budget = CudaMemory().budget()
    assert budget == MemoryBudget(6 * GIB, 16 * GIB, measured=True)


def test_cuda_failed_query_is_a_logged_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing ``mem_get_info`` falls back to total minus reserved, logged."""

    def broken(index: int) -> tuple[int, int]:
        raise RuntimeError(f"no NVML on device {index}")

    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 2 * GIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 1 * GIB)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=8 * GIB)
    )
    with caplog.at_level(logging.WARNING, logger="lemtrain.hardware.device_memory"):
        budget = CudaMemory().budget()
    assert budget == MemoryBudget(6 * GIB, 8 * GIB, measured=False)
    assert "estimating free memory" in caplog.text


def test_xpu_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """XPU reads ``torch.xpu.mem_get_info`` plus its reusable cache."""
    monkeypatch.setattr(torch.xpu, "mem_get_info", lambda index: (5 * GIB, 12 * GIB))
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda index: 2 * GIB)
    monkeypatch.setattr(torch.xpu, "memory_allocated", lambda index: 1 * GIB)
    assert XpuMemory().budget() == MemoryBudget(6 * GIB, 12 * GIB, measured=True)


def test_mps_budget_is_a_logged_conservative_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MPS: min(recommended working set - allocated, host available RAM), logged."""
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 20 * GIB)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 4 * GIB)
    monkeypatch.setattr(
        device_memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * GIB)
    )
    with caplog.at_level(logging.INFO, logger="lemtrain.hardware.device_memory"):
        budget = MpsMemory().budget()
    assert budget == MemoryBudget(10 * GIB, 20 * GIB, measured=False)
    assert "unified memory" in caplog.text


def test_mps_peak_is_sampled_at_observation_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a peak statistic the adapter keeps the highest observed allocation."""
    readings = iter([1 * GIB, 3 * GIB, 2 * GIB])
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: next(readings))
    memory = MpsMemory()
    memory.reset_peak()
    memory.observe()
    memory.observe()
    assert memory.peak_allocated() == 3 * GIB


def test_directml_budget_from_config_or_default(caplog: pytest.LogCaptureFixture) -> None:
    """DirectML has no query: the configured or documented default estimate, logged."""
    config = {"hardware": {"directml_assumed_free_gb": 6}}
    with caplog.at_level(logging.INFO, logger="lemtrain.hardware.device_memory"):
        configured = device_memory_for(torch.device("privateuseone"), config).budget()
    assert configured == MemoryBudget(6 * GIB, 6 * GIB, measured=False)
    assert "assuming 6.00 GB free" in caplog.text
    default = device_memory_for(torch.device("privateuseone")).budget()
    assert default is not None
    assert default.free_bytes == int(DIRECTML_ASSUMED_FREE_GB * GIB)


# --------------------------------------------------------------------- vision probe


def _audit(monkeypatch: pytest.MonkeyPatch, memory: FakeMemory) -> int:
    monkeypatch.setattr(probe, "device_memory_for", lambda device, config=None: memory)
    return audit_hardware_vram(
        "test_classifier",
        {"dataset_type": "classification", "input_size": 32},
        {},
        _Conv(),
        VramProbe(torch.device("cpu"), res_override=32),
    )


def test_vision_probe_runs_on_a_non_cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU backend other than CUDA is measured, not handed the configured default."""
    small = _audit(monkeypatch, FakeMemory("mps", MemoryBudget(GIB, 8 * GIB, False), 64 * 2**20))
    large = _audit(
        monkeypatch, FakeMemory("mps", MemoryBudget(8 * GIB, 8 * GIB, False), 64 * 2**20)
    )
    assert 1 <= small < large


def test_vision_probe_without_allocation_stats_uses_the_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DirectML-like backends report no peak; the resolution-scaled formula sizes a sample."""
    memory = FakeMemory("dml", MemoryBudget(2 * GIB, 2 * GIB, False), 0)
    batch = _audit(monkeypatch, memory)
    assert batch >= 1
    assert "reset_peak" in memory.calls


def test_cpu_audit_returns_the_configured_default() -> None:
    """The CPU adapter has no budget, so the vision audit keeps the configured batch."""
    batch = audit_hardware_vram(
        "test_classifier",
        {"dataset_type": "classification"},
        {"defaults": {"batch_size": 12}},
        _Conv(),
        VramProbe(torch.device("cpu")),
    )
    assert batch == 12


# --------------------------------------------------------------------- refit and clamp


def _budget_ctx() -> Any:
    model_info = {
        "dataset_type": "quality",
        "input_size": 256,
        "batch_size": 12,
        "accumulation_steps": 2,
        "optimization": {"res_ladder": [256, 384], "initial_fraction": 0.2},
    }
    governance = fresh_governance(model_info)
    governance.governor.curriculum.current_res = 384
    return SimpleNamespace(
        model_name="budget_model",
        governance=governance,
        models=SimpleNamespace(raw_model=_Conv(), model=None),
        hardware=SimpleNamespace(device_info=SimpleNamespace(device=torch.device("mps"))),
        run=SimpleNamespace(config={}, model_info=model_info, args=None),
    )


def test_resolution_refit_runs_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The jump refit probes a non-CUDA GPU and keeps the effective batch exact."""
    fake = FakeMemory("mps", MemoryBudget(GIB, 8 * GIB, False), 0)
    monkeypatch.setattr(batch_budget, "device_memory_for", lambda device, config=None: fake)
    seen: list[int] = []

    def audit(**kwargs: Any) -> int:
        seen.append(kwargs["probe"].res_override)
        return 5

    monkeypatch.setattr(batch_budget, "audit_hardware_vram", audit)
    ctx = _budget_ctx()
    assert batch_budget.refit_batch_for_resolution(ctx)
    governor = ctx.governance.governor
    assert seen == [384]
    assert (governor.current_batch, governor.current_acc) == (4, 6)


def _spec(device: torch.device) -> RunSpec:
    return RunSpec(
        args=argparse.Namespace(),
        config={},
        model_info={"dataset_type": "quality"},
        model_key="budget_model",
        project_root=Path("."),
        task_type="quality",
        device=device,
    )


def _train_state() -> TrainState:
    governor = SimpleNamespace(
        current_batch=12, curriculum=CurriculumState(current_batch=12, current_acc=2)
    )
    return TrainState(
        raw_model=None,
        model=None,
        optimizer=None,
        scheduler=None,
        governor=governor,
        sota_tracker=None,
    )


def test_resume_clamp_on_xpu_keeps_effective_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resume clamp audits a non-CUDA GPU and fits by the exact-divisor rule."""
    fake = FakeMemory("xpu", MemoryBudget(GIB, 8 * GIB, True), 0)
    monkeypatch.setattr(context_resume, "device_memory_for", lambda device, config=None: fake)
    monkeypatch.setattr(context_resume, "audit_hardware_vram", lambda **kwargs: 5)
    train = _train_state()
    batch = _clamp_resumed_batch(_spec(torch.device("xpu")), train, (384, 0.4), 12)
    curriculum = train.governor.curriculum
    assert batch == 4
    assert curriculum.current_batch * curriculum.current_acc == 24


def test_resume_clamp_is_a_cpu_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a device budget the restored batch is kept and no audit runs."""

    def audit(**kwargs: Any) -> int:
        raise AssertionError(f"no audit on the CPU: {kwargs}")

    monkeypatch.setattr(context_resume, "audit_hardware_vram", audit)
    assert _clamp_resumed_batch(_spec(torch.device("cpu")), _train_state(), (384, 0.4), 12) == 12


# --------------------------------------------------------------------- OOM relief


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "HIP out of memory. Tried to allocate 1.00 GiB",
        "MPS backend out of memory (MPS allocated: 9.0 GB)",
        "XPU out of memory. Tried to allocate 512.00 MiB",
        "Could not allocate tensor with 8 bytes. There is not enough GPU video memory available!",
    ],
)
def test_out_of_memory_detection_per_backend(message: str) -> None:
    """Every backend's out-of-memory error is recognised; other errors are not."""
    assert is_out_of_memory(RuntimeError(message))
    assert not is_out_of_memory(RuntimeError("shape mismatch"))
    assert is_out_of_memory(torch.OutOfMemoryError("anything"))


def test_oom_relief_releases_every_present_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The governor's OOM relief flushes each present backend and halves exactly."""
    fakes = [FakeMemory("mps", None, 0), FakeMemory("xpu", None, 0)]
    monkeypatch.setattr(device_memory, "present_device_memories", lambda: fakes)
    release_device_memory("test")
    assert all(fake.calls == ["release_cache"] for fake in fakes)

    governor = fresh_governance({"batch_size": 12, "accumulation_steps": 2}).governor
    new_batch, new_acc = governor.handle_oom(current_batch=12)
    assert (new_batch, new_acc) == (6, 4)
    assert all(fake.calls == ["release_cache", "release_cache"] for fake in fakes)
