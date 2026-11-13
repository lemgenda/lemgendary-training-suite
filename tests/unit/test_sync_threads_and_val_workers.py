"""Bounded exit wait for cloud uploads and persistent validation workers (R2-T9)."""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
import torch
from torch.utils.data import TensorDataset

from training.cloud.sync_threads import (
    DEFAULT_EXIT_TIMEOUT_SEC,
    SyncThreadRegistry,
    exit_timeout,
)
from training.data.loaders import (
    LoaderPlacement,
    ValLoaderOptions,
    WorkerOverrides,
    build_val_loader,
)


def test_exit_timeout_reads_the_fleet_config() -> None:
    """Default 300 s; ``fleet.sync_exit_timeout_sec`` overrides; negatives are rejected."""
    assert exit_timeout(None) == DEFAULT_EXIT_TIMEOUT_SEC == 300.0
    assert exit_timeout({"fleet": {"sync_exit_timeout_sec": 12}}) == 12.0
    with pytest.raises(ValueError):
        exit_timeout({"fleet": {"sync_exit_timeout_sec": -1}})


def test_finished_uploads_are_joined_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Uploads run on daemon threads and a finished one is reported at exit."""
    registry = SyncThreadRegistry()
    done = threading.Event()
    thread = registry.start(done.set, "upload-ok", timeout=5.0)
    assert thread.daemon
    with caplog.at_level(logging.INFO, logger="lemtrain.cloud_sync"):
        thread.join(5.0)
        assert registry.join_pending() == (0, 0)
    assert done.is_set()


def test_a_stalled_upload_is_abandoned_after_the_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exit wait is bounded and names the abandoned upload."""
    registry = SyncThreadRegistry()
    release = threading.Event()
    registry.start(lambda: release.wait(10.0), "upload-stalled", timeout=0.05)
    with caplog.at_level(logging.WARNING, logger="lemtrain.cloud_sync"):
        finished, abandoned = registry.join_pending()
    release.set()
    assert (finished, abandoned) == (0, 1)
    assert "upload-stalled" in caplog.text


def _val_loader(**overrides: Any):
    dataset = TensorDataset(torch.zeros(8, 2), torch.zeros(8, 1))
    return build_val_loader(
        dataset,
        batch_size=4,
        options=ValLoaderOptions(workers=WorkerOverrides(**overrides)),
        placement=LoaderPlacement(device=torch.device("cpu"), env="local", config={}),
    )


def test_validation_workers_persist_when_workers_run() -> None:
    """With workers the validation loader keeps them alive unless told otherwise."""
    assert _val_loader(num_workers=1, persistent_workers=True).persistent_workers
    assert not _val_loader(num_workers=1, persistent_workers=False).persistent_workers
    assert not _val_loader(num_workers=0, persistent_workers=True).persistent_workers
