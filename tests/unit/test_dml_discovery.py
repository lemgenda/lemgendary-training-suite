"""DirectML discovery through the ``torch_directml`` module (R4-FU item 6).

torch-directml 0.2.x has no ``torch.dml`` attribute: the backend is the ``torch_directml``
module (``is_available``, ``device_count``, ``device_name``, ``default_device``, ``device``)
and its devices are ``privateuseone:<index>``. The module is mocked; no DirectML is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training.hardware import discovery
from training.hardware.discovery import discover_device, dml_device_info

_NAMES = ["AMD Radeon RX 6600", "Intel UHD Graphics 770"]


def _fake_directml(available: bool = True, default: int = 0) -> SimpleNamespace:
    """The torch_directml functions the probe calls."""
    return SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(_NAMES) if available else 0,
        device_name=lambda index: _NAMES[index],
        default_device=lambda: default,
        device=lambda index=None: torch.device(
            "privateuseone", default if index is None else index
        ),
    )


def _install(monkeypatch: pytest.MonkeyPatch, loader) -> None:
    monkeypatch.setattr(discovery, "load_torch_directml", loader)


def test_directml_device_comes_from_torch_directml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default DirectML adapter becomes a ``privateuseone`` device with every name."""
    _install(monkeypatch, lambda: _fake_directml(default=1))
    info = dml_device_info()
    assert info is not None
    assert info.device == torch.device("privateuseone", 1)
    assert info.device_type == "dml"
    assert info.device_count == 2
    assert info.device_names == _NAMES
    assert not info.is_cuda


def test_unavailable_directml_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DirectML adapter (driver missing): the probe yields nothing."""
    _install(monkeypatch, lambda: _fake_directml(available=False))
    assert dml_device_info() is None


@pytest.mark.parametrize("error", [ImportError("no torch_directml"), OSError("DLL load failed")])
def test_missing_or_broken_backend_is_skipped(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """torch-directml not installed, or its DLL not loadable against this torch."""

    def _raise() -> SimpleNamespace:
        raise error

    _install(monkeypatch, _raise)
    assert dml_device_info() is None


def test_auto_discovery_selects_directml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without CUDA, MPS or XPU, ``discover_device`` returns the DirectML device."""
    _install(monkeypatch, _fake_directml)
    monkeypatch.setattr(discovery, "_ACCELERATOR_PROBES", (dml_device_info,))
    monkeypatch.setattr(discovery, "_CACHED_DEVICE_INFO", {})
    info = discover_device()
    assert info.device.type == "privateuseone"
    assert info.device_names[0] == _NAMES[0]


def test_torch_has_no_dml_attribute() -> None:
    """The old probe read ``torch.dml``, which torch-directml never defines."""
    assert not hasattr(torch, "dml")
