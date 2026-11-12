"""Capture and restore the random number generators a resumed run continues from.

A checkpoint records the Python ``random`` state, the legacy NumPy global state, the torch
CPU generator and every present device generator (CUDA, which covers ROCm builds, MPS and
XPU). Restoring them makes the training loop's own randomness (dropout, stochastic depth,
random masking) continue where the interrupted run stopped. Sample augmentation does not
depend on this state: :mod:`training.data.sample_seed` derives it per sample.

The record holds only lists, ints, floats, strings and tensors, so it also loads under
``torch.load(weights_only=True)``. Checkpoints written before the record existed resume
with fresh generators, which is logged.

Under DDP every rank has its own generators (dropout masks differ per rank), so a checkpoint
also holds ``rng_state_ranks``: the record of every rank, gathered with one collective
``all_gather_object``. A resumed rank restores its own entry; a checkpoint with only
``rng_state`` (single process, or written before per-rank records) restores that one on every
rank, and a rank beyond the recorded list (more processes than the run that saved) does too.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger("lemtrain.rng_state")

PAYLOAD_KEY = "rng_state"
"""Checkpoint payload key of the generator record (rank 0, or the only process)."""
RANKS_PAYLOAD_KEY = "rng_state_ranks"
"""Checkpoint payload key of the per-rank generator records (index = DDP rank)."""

RNG_RESTORE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
)
"""Failure modes of restoring a malformed or foreign generator record."""


@dataclass(frozen=True)
class DeviceGenerators:
    """How to read and write one accelerator backend's per-device generator states."""

    name: str
    available: Callable[[], bool]
    get_all: Callable[[], list[torch.Tensor]]
    set_one: Callable[[torch.Tensor, int], None]


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _cuda_get_all() -> list[torch.Tensor]:
    return list(torch.cuda.get_rng_state_all())


def _cuda_set_one(state: torch.Tensor, index: int) -> None:
    torch.cuda.set_rng_state(state, device=index)


def _mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def _mps_get_all() -> list[torch.Tensor]:
    return [torch.mps.get_rng_state()]


def _mps_set_one(state: torch.Tensor, index: int) -> None:
    if index == 0:
        torch.mps.set_rng_state(state)


def _xpu_module() -> Any:
    return getattr(torch, "xpu", None)


def _xpu_available() -> bool:
    xpu = _xpu_module()
    return xpu is not None and bool(xpu.is_available())


def _xpu_get_all() -> list[torch.Tensor]:
    return list(_xpu_module().get_rng_state_all())


def _xpu_set_one(state: torch.Tensor, index: int) -> None:
    _xpu_module().set_rng_state(state, device=index)


DEVICE_GENERATORS: tuple[DeviceGenerators, ...] = (
    DeviceGenerators("cuda", _cuda_available, _cuda_get_all, _cuda_set_one),
    DeviceGenerators("mps", _mps_available, _mps_get_all, _mps_set_one),
    DeviceGenerators("xpu", _xpu_available, _xpu_get_all, _xpu_set_one),
)
"""The device backends whose generators a checkpoint records, when present."""


def capture_host_rng() -> dict[str, Any]:
    """The Python, NumPy and torch CPU generator states (weights-only loadable)."""
    py_version, py_internal, py_gauss = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    return {
        "python": [int(py_version), [int(v) for v in py_internal], py_gauss],
        "numpy": {
            "name": str(np_name),
            "keys": torch.from_numpy(np.asarray(np_keys, dtype=np.int64)),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch_cpu": torch.get_rng_state(),
    }


def restore_host_rng(record: dict[str, Any]) -> None:
    """Set the Python, NumPy and torch CPU generators from :func:`capture_host_rng` output."""
    py_version, py_internal, py_gauss = record["python"]
    random.setstate((int(py_version), tuple(int(v) for v in py_internal), py_gauss))
    np_state = record["numpy"]
    keys = torch.as_tensor(np_state["keys"]).cpu().numpy().astype(np.uint32)
    np.random.set_state(
        (
            str(np_state["name"]),
            keys,
            int(np_state["pos"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(record["torch_cpu"]).cpu())


def capture_rng_state() -> dict[str, Any]:
    """The full generator record of this process: host plus every present device backend."""
    devices = {
        backend.name: [state.cpu() for state in backend.get_all()]
        for backend in DEVICE_GENERATORS
        if backend.available()
    }
    return {**capture_host_rng(), "devices": devices}


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def capture_rank_rng_states() -> list[dict[str, Any]]:
    """The generator record of every DDP rank (index = rank); this process's alone without DDP.

    A collective call under DDP: every rank must call it at the same point.
    """
    own = capture_rng_state()
    if not _distributed():
        return [own]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, own)
    return [record for record in gathered if isinstance(record, dict)]


def rng_payload() -> dict[str, Any]:
    """Payload entries of the generator state: ``rng_state`` and ``rng_state_ranks``.

    A collective call under DDP (see :func:`capture_rank_rng_states`).
    """
    ranks = capture_rank_rng_states()
    return {PAYLOAD_KEY: ranks[0], RANKS_PAYLOAD_KEY: ranks}


def _record_for_this_rank(payload: dict[str, Any]) -> Any:
    """This rank's record from ``rng_state_ranks``, else the single ``rng_state`` record."""
    ranks = payload.get(RANKS_PAYLOAD_KEY)
    rank = dist.get_rank() if _distributed() else 0
    if isinstance(ranks, list) and rank < len(ranks) and isinstance(ranks[rank], dict):
        return ranks[rank]
    if isinstance(ranks, list) and ranks:
        logger.info(
            "[RNG] Checkpoint has generator states for %d rank(s); rank %d restores rank 0's.",
            len(ranks),
            rank,
        )
    return payload.get(PAYLOAD_KEY)


def _restore_devices(saved: dict[str, Any]) -> list[str]:
    """Restore the recorded device generators that this host also has; log the rest."""
    restored: list[str] = []
    for backend in DEVICE_GENERATORS:
        states = saved.get(backend.name)
        if not states:
            continue
        if not backend.available():
            logger.info(
                "[RNG] Checkpoint has %s generator state but this host has no %s device; "
                "it starts fresh.",
                backend.name,
                backend.name,
            )
            continue
        present = len(backend.get_all())
        if present != len(states):
            logger.info(
                "[RNG] Checkpoint has %d %s generator state(s), this host %d; restoring %d.",
                len(states),
                backend.name,
                present,
                min(present, len(states)),
            )
        for index, state in enumerate(states[:present]):
            backend.set_one(torch.as_tensor(state).cpu(), index)
        restored.append(backend.name)
    return restored


def restore_rng_state(payload: dict[str, Any]) -> bool:
    """Restore this rank's generators from a checkpoint ``payload``; return whether it had one.

    Each DDP rank restores its own record (``rng_state_ranks``); a single-process checkpoint
    restores ``rng_state``. A checkpoint without a record (written before it existed) or with
    an unreadable one keeps the current generators, which is logged.
    """
    record = _record_for_this_rank(payload)
    if not isinstance(record, dict):
        logger.info(
            "[RNG] Checkpoint has no generator state (written before it was recorded); "
            "the resumed run continues with fresh generators."
        )
        return False
    try:
        restore_host_rng(record)
        devices = _restore_devices(dict(record.get("devices") or {}))
    except RNG_RESTORE_ERRORS as rng_err:
        logger.warning(
            "[RNG] Checkpoint generator state could not be restored (%s); "
            "the resumed run continues with fresh generators.",
            rng_err,
        )
        return False
    logger.info(
        "[RNG] Restored python, numpy and torch CPU generators%s.",
        f" and {', '.join(devices)} device generators" if devices else "",
    )
    return True
