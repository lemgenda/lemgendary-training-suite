"""Cross-platform helpers: venv site-packages, DDP backend, spawn sizing, process groups."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest
import torch

from training.data import workers
from training.parallel import ddp
from training.utils import interrupt
from training.utils.paths import venv_site_packages


def test_venv_site_packages_matches_the_running_venv() -> None:
    """The helper finds this interpreter's own site-packages from its venv root."""
    site = venv_site_packages(Path(sys.prefix))
    assert site.name == "site-packages"
    assert site.exists() == (sys.prefix != sys.base_prefix)


def test_venv_site_packages_layout_per_platform() -> None:
    """Windows uses ``Lib/site-packages``; POSIX ``lib/pythonX.Y/site-packages``."""
    parts = venv_site_packages(Path("venv-root")).parts
    if sys.platform == "win32":
        assert parts[-2:] == ("Lib", "site-packages")
    else:
        assert parts[-3:] == (
            "lib",
            f"python{sys.version_info[0]}.{sys.version_info[1]}",
            "site-packages",
        )


@pytest.mark.parametrize(
    ("platform", "cuda", "nccl", "expected"),
    [
        ("linux", True, True, "nccl"),
        ("linux", True, False, "gloo"),
        ("linux", False, True, "gloo"),
        ("win32", True, True, "gloo"),
    ],
)
def test_ddp_backend_selection(
    monkeypatch: pytest.MonkeyPatch, platform: str, cuda: bool, nccl: bool, expected: str
) -> None:
    """NCCL only with CUDA and NCCL off Windows; Gloo everywhere else."""
    monkeypatch.setattr(ddp.sys, "platform", platform)
    monkeypatch.setattr(ddp.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(ddp.dist, "is_nccl_available", lambda: nccl)
    assert ddp.select_backend() == expected


def test_macos_sizes_workers_like_windows_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS spawns workers like Windows, so it gets the spawn-aware worker budget."""
    monkeypatch.setattr(workers, "_system_ram_gb", lambda: 6.0)
    monkeypatch.setattr(workers, "_available_ram_gb", lambda: 3.0)
    monkeypatch.setattr(workers, "_set_cpu_threads", lambda env, cpu_count: 1)
    monkeypatch.setattr(workers.os, "cpu_count", lambda: 16)
    sizes = {}
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(workers.sys, "platform", platform)
        topology = workers.compute_worker_topology(env="local", device=torch.device("cpu"))
        sizes[platform] = topology.num_workers
    assert sizes["darwin"] == sizes["win32"] == 0
    assert sizes["linux"] == 2


class _FakeProcess:
    """Stands in for a running ``Popen`` whose termination is recorded."""

    pid = 4242

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> None:
        """Report the process as still running."""
        return

    def terminate(self) -> None:
        """Record that the process alone was terminated."""
        self.terminated = True


@pytest.mark.parametrize("group_leader", [True, False])
def test_posix_cleanup_stops_session_process_groups(
    monkeypatch: pytest.MonkeyPatch, group_leader: bool
) -> None:
    """A child started with ``start_new_session`` is stopped with its whole process group."""
    killed: list[tuple[int, int]] = []
    proc = _FakeProcess()

    def fake_kill(pid: int, sig: int) -> None:
        if not group_leader:
            raise ProcessLookupError(pid)
        killed.append((pid, sig))

    monkeypatch.setattr(interrupt.sys, "platform", "linux")
    monkeypatch.setattr(interrupt.os, "kill", fake_kill)
    monkeypatch.setattr(interrupt, "_ACTIVE_PROCESSES", [proc])
    interrupt.cleanup_active_processes()
    assert killed == ([(-proc.pid, signal.SIGTERM)] if group_leader else [])
    assert proc.terminated is not group_leader
