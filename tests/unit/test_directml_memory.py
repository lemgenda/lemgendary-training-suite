"""DirectML free memory from the DXGI budget and its fallback chain (R4-C6).

The COM source is replaced by a fake (no GPU needed); only the last test reads the real
DXGI, on Windows, and checks nothing but structural sanity.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from training.hardware import device_memory, dxgi_memory
from training.hardware.device_memory import (
    DIRECTML_ASSUMED_FREE_GB,
    DIRECTML_DEDICATED_FRACTION,
    DirectMLMemory,
    MemoryBudget,
    device_memory_for,
)
from training.hardware.dxgi_memory import (
    AdapterMemory,
    DxgiQueryError,
    candidate_adapters,
    is_unified_memory,
    read_dxgi_adapters,
)

GIB = 1024**3

RTX = AdapterMemory("NVIDIA RTX", 8 * GIB, 16 * GIB, False, 7 * GIB, 1 * GIB)
IGPU = AdapterMemory("AMD Radeon iGPU", GIB // 4, 16 * GIB, False, 40 * GIB, 0)
WARP = AdapterMemory("Microsoft Basic Render Driver", 0, 16 * GIB, True, None, None)


@dataclass
class FakeReader:
    """An adapter reader returning a fixed list, or raising ``error``; counts its reads."""

    adapters: list[AdapterMemory]
    error: Exception | None = None
    reads: int = 0

    def __call__(self) -> list[AdapterMemory]:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return list(self.adapters)


def _dml(reader: FakeReader | None, name: str | None = None) -> DirectMLMemory:
    return DirectMLMemory(0, read_adapters=reader, name_of=lambda _index: name)


def _host_available(monkeypatch: pytest.MonkeyPatch, available: int) -> None:
    monkeypatch.setattr(
        device_memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )


def test_dxgi_budget_minus_usage_is_measured_and_reread() -> None:
    """Free is budget minus current usage, total the budget; every call reads DXGI again."""
    reader = FakeReader([RTX, WARP])
    memory = _dml(reader, "NVIDIA RTX")
    assert memory.budget() == MemoryBudget(6 * GIB, 7 * GIB, measured=True)
    memory.budget()
    assert reader.reads == 2
    assert memory.device_name() == "NVIDIA RTX"


def test_usage_above_budget_leaves_no_free_memory() -> None:
    """Usage above the budget is clamped to zero free memory, still measured."""
    over = AdapterMemory("NVIDIA RTX", 8 * GIB, 0, False, 4 * GIB, 5 * GIB)
    assert _dml(FakeReader([over]), "NVIDIA RTX").budget() == MemoryBudget(0, 4 * GIB, True)


def test_integrated_gpu_is_capped_by_host_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    """An integrated GPU's local segment is host RAM: free is capped by available RAM."""
    _host_available(monkeypatch, 12 * GIB)
    budget = _dml(FakeReader([IGPU]), "AMD Radeon iGPU").budget()
    assert budget == MemoryBudget(12 * GIB, 40 * GIB, measured=True)


def test_unified_memory_detection() -> None:
    """Under 1 GiB dedicated, or a budget above dedicated plus shared, is unified memory."""
    assert is_unified_memory(IGPU)
    assert not is_unified_memory(RTX)
    large_budget = AdapterMemory("APU", 2 * GIB, 4 * GIB, False, 10 * GIB, 0)
    assert is_unified_memory(large_budget)


def test_name_selects_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The torch-directml name picks its DXGI adapter even when another is smaller."""
    _host_available(monkeypatch, 64 * GIB)
    budget = _dml(FakeReader([IGPU, RTX]), "AMD Radeon iGPU").budget()
    assert budget is not None
    assert budget.total_bytes == 40 * GIB


def test_ambiguous_or_unknown_name_takes_the_smallest_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two adapters with the device name, or none, leave the smallest budget (logged)."""
    twin = AdapterMemory("NVIDIA RTX", 12 * GIB, 16 * GIB, False, 11 * GIB, 0)
    with caplog.at_level(logging.INFO, logger="lemtrain.hardware.dxgi_memory"):
        ambiguous = _dml(FakeReader([twin, RTX]), "NVIDIA RTX").budget()
        unmatched = _dml(FakeReader([twin, RTX]), "Some Other GPU").budget()
        unnamed = _dml(FakeReader([twin, RTX]), None).budget()
    for budget in (ambiguous, unmatched, unnamed):
        assert budget == MemoryBudget(6 * GIB, 7 * GIB, measured=True)
    assert "2 DXGI adapters are named 'NVIDIA RTX'" in caplog.text
    assert "No DXGI adapter is named 'Some Other GPU'" in caplog.text


def test_software_adapter_is_skipped() -> None:
    """The Basic Render Driver is never a candidate, even when it carries the name."""
    assert candidate_adapters([WARP, RTX], "Microsoft Basic Render Driver") == [RTX]
    assert candidate_adapters([WARP], None) == []
    assert _dml(FakeReader([WARP]), None).budget() == MemoryBudget(
        int(DIRECTML_ASSUMED_FREE_GB * GIB), int(DIRECTML_ASSUMED_FREE_GB * GIB), False
    )


def test_no_dxgi_budget_falls_back_to_dedicated_memory() -> None:
    """Without IDXGIAdapter3 the estimate is a fraction of dedicated video memory."""
    old = AdapterMemory("Old GPU", 4 * GIB, 8 * GIB, False, None, None)
    budget = _dml(FakeReader([old]), "Old GPU").budget()
    assert budget == MemoryBudget(int(4 * GIB * DIRECTML_DEDICATED_FRACTION), 4 * GIB, False)


@pytest.mark.parametrize(
    "error", [DxgiQueryError("no dxgi"), OSError("COM"), AttributeError("x"), ValueError("y")]
)
def test_dxgi_failure_falls_back_to_the_assumed_default(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed DXGI read (no adapters, no description) ends at the documented default."""
    with caplog.at_level(logging.WARNING, logger="lemtrain.hardware.device_memory"):
        budget = _dml(FakeReader([], error), "NVIDIA RTX").budget()
    default = int(DIRECTML_ASSUMED_FREE_GB * GIB)
    assert budget == MemoryBudget(default, default, measured=False)
    assert "DXGI query failed" in caplog.text


def test_explicit_config_wins_over_dxgi(monkeypatch: pytest.MonkeyPatch) -> None:
    """A set ``hardware.directml_assumed_free_gb`` is used without reading DXGI."""
    reader = FakeReader([RTX])
    monkeypatch.setattr(device_memory, "default_adapter_reader", lambda: reader)
    config = {"hardware": {"directml_assumed_free_gb": 3}}
    budget = device_memory_for(torch.device("privateuseone"), config).budget()
    assert budget == MemoryBudget(3 * GIB, 3 * GIB, measured=False)
    assert reader.reads == 0
    unset = device_memory_for(torch.device("privateuseone"), {"hardware": {}}).budget()
    assert unset is not None and unset.measured


def test_non_windows_skips_dxgi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off Windows there is no DXGI source and the real source refuses to run."""
    monkeypatch.setattr(dxgi_memory.sys, "platform", "linux")
    assert dxgi_memory.default_adapter_reader() is None
    with pytest.raises(DxgiQueryError):
        read_dxgi_adapters()


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI exists only on Windows")
def test_real_dxgi_query_is_structurally_sane() -> None:
    """The real query returns non-negative sizes (no assumption about the installed GPU)."""
    try:
        adapters = read_dxgi_adapters()
    except DxgiQueryError as exc:
        pytest.skip(f"DXGI unavailable on this machine: {exc}")
    for adapter in adapters:
        assert adapter.dedicated_bytes >= 0
        assert adapter.shared_bytes >= 0
        assert adapter.budget_bytes is None or adapter.budget_bytes >= 0
        assert adapter.usage_bytes is None or adapter.usage_bytes >= 0
