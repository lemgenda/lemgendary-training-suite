"""Per-process GPU memory budget from DXGI, the source of DirectML free memory (Windows).

``torch-directml`` has no memory query. DXGI has one: ``IDXGIAdapter3::QueryVideoMemoryInfo``
on the LOCAL segment returns the operating system's **Budget** for this process and the
process's **CurrentUsage** (which counts torch-directml's own allocations). The budget
shrinks when other applications use video memory, so it is read again on every call.

The COM calls use only the standard library (``ctypes``): ``CreateDXGIFactory1``, then per
adapter ``IDXGIFactory1::EnumAdapters1`` (vtable slot 12), ``IDXGIAdapter1::GetDesc1``
(slot 10), ``QueryInterface`` for ``IDXGIAdapter3`` (slot 0) and ``QueryVideoMemoryInfo``
(slot 14); every interface is released (slot 2). ``DXGI_ADAPTER_DESC1`` and
``DXGI_QUERY_VIDEO_MEMORY_INFO`` are read from raw buffers with :mod:`struct`. The
Windows-only code runs behind ``sys.platform == "win32"`` checks; elsewhere
:func:`default_adapter_reader` returns ``None`` and the DirectML adapter skips DXGI.

Consumers depend on the :data:`AdapterReader` type and the pure helpers
(:func:`candidate_adapters`, :func:`is_unified_memory`), so tests replace the COM reader
with a fake.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("lemtrain.hardware.dxgi_memory")

_GIB = 1024**3

UMA_DEDICATED_LIMIT_BYTES = _GIB
"""An adapter with less dedicated video memory than this is integrated (unified memory)."""

_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0

_VTABLE_QUERY_INTERFACE = 0
_VTABLE_RELEASE = 2
_VTABLE_GET_DESC1 = 10
_VTABLE_ENUM_ADAPTERS1 = 12
_VTABLE_QUERY_VIDEO_MEMORY_INFO = 14

_IID_IDXGI_FACTORY1 = uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_IDXGI_ADAPTER3 = uuid.UUID("645967a4-1392-4310-a798-8053ce3e93fd")

_DESC1_LAYOUT = struct.Struct("@256s4I3NIiI")
"""``DXGI_ADAPTER_DESC1``: Description (WCHAR[128]), VendorId, DeviceId, SubSysId,
Revision, DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory (SIZE_T),
AdapterLuid (DWORD, LONG), Flags."""

_MEMORY_INFO_LAYOUT = struct.Struct("@4Q")
"""``DXGI_QUERY_VIDEO_MEMORY_INFO``: Budget, CurrentUsage, AvailableForReservation,
CurrentReservation."""

_BUFFER_PADDING = 64
"""Spare bytes after each output buffer (trailing structure padding)."""


class DxgiQueryError(OSError):
    """DXGI could not be loaded or queried (no DirectX 12 runtime, not Windows, COM error)."""


@dataclass(frozen=True)
class AdapterMemory:
    """One DXGI adapter: its description and, when ``IDXGIAdapter3`` answered, its budget.

    ``budget_bytes`` and ``usage_bytes`` are ``None`` when the adapter does not implement
    ``IDXGIAdapter3`` or the query failed; the dedicated size still serves as an estimate.
    """

    name: str
    dedicated_bytes: int
    shared_bytes: int
    software: bool
    budget_bytes: int | None
    usage_bytes: int | None


AdapterReader = Callable[[], list[AdapterMemory]]
"""Reads every adapter of the machine (software ones flagged); raises :class:`OSError`
(:class:`DxgiQueryError`) when the query fails."""


def is_unified_memory(adapter: AdapterMemory) -> bool:
    """Whether ``adapter`` is an integrated GPU whose local segment is host RAM.

    True when its dedicated memory is under :data:`UMA_DEDICATED_LIMIT_BYTES`, or when its
    LOCAL budget exceeds dedicated plus shared memory (only system RAM can back that).
    """
    if adapter.dedicated_bytes < UMA_DEDICATED_LIMIT_BYTES:
        return True
    budget = adapter.budget_bytes
    return budget is not None and budget > adapter.dedicated_bytes + adapter.shared_bytes


def candidate_adapters(
    adapters: list[AdapterMemory], device_name: str | None
) -> list[AdapterMemory]:
    """The hardware adapters that may be the torch-directml device named ``device_name``.

    Software adapters are skipped. Exactly one adapter whose description equals the name is
    the device; an ambiguous or unmatched name (or no name) leaves every matching (or every
    hardware) adapter as a candidate, and the caller takes the smallest. Logged.
    """
    hardware = [adapter for adapter in adapters if not adapter.software]
    if device_name is None:
        logger.info(
            "[VRAM] torch-directml device name unknown; using the smallest of %d DXGI "
            "hardware adapter(s).",
            len(hardware),
        )
        return hardware
    wanted = device_name.strip()
    matched = [adapter for adapter in hardware if adapter.name.strip() == wanted]
    if len(matched) == 1:
        return matched
    if matched:
        logger.info(
            "[VRAM] %d DXGI adapters are named %r; using the smallest budget among them.",
            len(matched),
            wanted,
        )
        return matched
    logger.info(
        "[VRAM] No DXGI adapter is named %r; using the smallest of %d hardware adapter(s).",
        wanted,
        len(hardware),
    )
    return hardware


def _guid(iid: uuid.UUID) -> ctypes.Array[ctypes.c_char]:
    """``iid`` in the in-memory GUID layout COM expects."""
    return ctypes.create_string_buffer(iid.bytes_le, 16)


def _com_method(obj: ctypes.c_void_p, slot: int, *argtypes: Any) -> Callable[..., Any]:
    """HRESULT method ``slot`` of the COM object ``obj``'s vtable, bound to ``obj``.

    A failed HRESULT raises :class:`OSError` (``winerror`` holds the code).
    """
    if sys.platform != "win32":
        raise DxgiQueryError("COM calls exist only on Windows")
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    function = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[slot])

    def bound(*args: Any) -> Any:
        return function(obj, *args)

    return bound


def _release(obj: ctypes.c_void_p) -> None:
    """``IUnknown::Release`` of ``obj`` when it holds an interface."""
    if not obj.value or sys.platform != "win32":
        return
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[_VTABLE_RELEASE])(obj)


def _adapter_budget(adapter: ctypes.c_void_p, name: str) -> tuple[int | None, int | None]:
    """LOCAL budget and current usage of ``adapter`` (``None`` pair when unsupported)."""
    adapter3 = ctypes.c_void_p()
    iid = _guid(_IID_IDXGI_ADAPTER3)
    info = ctypes.create_string_buffer(_MEMORY_INFO_LAYOUT.size + _BUFFER_PADDING)
    try:
        _com_method(adapter, _VTABLE_QUERY_INTERFACE, ctypes.c_void_p, ctypes.c_void_p)(
            ctypes.addressof(iid), ctypes.addressof(adapter3)
        )
        query = _com_method(
            adapter3, _VTABLE_QUERY_VIDEO_MEMORY_INFO, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p
        )
        query(0, _DXGI_MEMORY_SEGMENT_GROUP_LOCAL, ctypes.addressof(info))
    except OSError as exc:
        logger.info("[VRAM] DXGI budget query failed for %r: %s", name, exc)
        return None, None
    finally:
        _release(adapter3)
    budget, usage, _available, _reserved = _MEMORY_INFO_LAYOUT.unpack_from(info.raw)
    return int(budget), int(usage)


def _describe_adapter(adapter: ctypes.c_void_p) -> AdapterMemory:
    """Description of ``adapter`` and, for a hardware adapter, its LOCAL budget."""
    desc = ctypes.create_string_buffer(_DESC1_LAYOUT.size + _BUFFER_PADDING)
    _com_method(adapter, _VTABLE_GET_DESC1, ctypes.c_void_p)(ctypes.addressof(desc))
    fields = _DESC1_LAYOUT.unpack_from(desc.raw)
    name = fields[0].decode("utf-16-le", errors="replace").split("\0", 1)[0]
    dedicated, shared, flags = int(fields[5]), int(fields[7]), int(fields[10])
    software = bool(flags & _DXGI_ADAPTER_FLAG_SOFTWARE)
    budget, usage = (None, None) if software else _adapter_budget(adapter, name)
    return AdapterMemory(name, dedicated, shared, software, budget, usage)


def _enumerate_adapters(factory: ctypes.c_void_p) -> list[AdapterMemory]:
    """Every adapter of ``factory`` until DXGI reports ``DXGI_ERROR_NOT_FOUND``."""
    enum_adapters = _com_method(factory, _VTABLE_ENUM_ADAPTERS1, ctypes.c_uint, ctypes.c_void_p)
    found: list[AdapterMemory] = []
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        try:
            enum_adapters(index, ctypes.addressof(adapter))
        except OSError as exc:
            if (getattr(exc, "winerror", None) or 0) & 0xFFFFFFFF == _DXGI_ERROR_NOT_FOUND:
                return found
            raise DxgiQueryError(f"EnumAdapters1({index}) failed: {exc}") from exc
        index += 1
        try:
            found.append(_describe_adapter(adapter))
        finally:
            _release(adapter)


def read_dxgi_adapters() -> list[AdapterMemory]:
    """Every adapter of the machine read through DXGI (an :data:`AdapterReader`, Windows)."""
    if sys.platform != "win32":
        raise DxgiQueryError("DXGI exists only on Windows")
    factory = ctypes.c_void_p()
    iid = _guid(_IID_IDXGI_FACTORY1)
    try:
        create_factory = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create_factory.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        create_factory.restype = ctypes.c_long
        result = int(create_factory(ctypes.addressof(iid), ctypes.addressof(factory)))
    except (OSError, AttributeError) as exc:
        raise DxgiQueryError(f"dxgi.dll is unavailable: {exc}") from exc
    if result:
        raise DxgiQueryError(f"CreateDXGIFactory1 failed (HRESULT 0x{result & 0xFFFFFFFF:08X})")
    try:
        return _enumerate_adapters(factory)
    finally:
        _release(factory)


def default_adapter_reader() -> AdapterReader | None:
    """The DXGI reader on Windows, ``None`` on every other platform (no DXGI there)."""
    if sys.platform == "win32":
        return read_dxgi_adapters
    return None
