"""Zero-rebuild curriculum transitions over WebDataset shards (audit 7.1.4).

A fraction or resolution change re-slices the cached sample list and a rebuilt
loader's workers read by byte offset, so no transition rescans tar headers. The
fraction subsets are nested, so a larger fraction never drops a sample that a
smaller one trained on (Monotonic Ratchet).
"""

from __future__ import annotations

import itertools
import pickle
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.dataset import MultiTaskDataset
from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar
from training.data.containers import tar_offsets

MODEL_KEY = "nafnet_denoising"
SAMPLES = 20


def _dataset(tmp_path: Path) -> MultiTaskDataset:
    members: dict[str, bytes] = {}
    for i in range(SAMPLES):
        key = f"s{i:03d}"
        members[f"{key}.webp"] = webp_bytes((i * 10, 90, 160), size=(24, 24), lossless=True)
        members[f"{key}.target.webp"] = webp_bytes((200, 210, i * 10), size=(24, 24), lossless=True)
        members[f"{key}.json"] = json_bytes({"task": "restoration"})
    data_root = tmp_path / "datasets"
    registry_path = Path(__file__).resolve().parents[2] / "unified_models_v2.yaml"
    config: dict[str, Any] = {
        "unified_models": "unified_models_v2.yaml",
        "paths": {"datasets_root": str(data_root)},
    }
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    shards = data_root / registry[MODEL_KEY]["datasets"][0] / "shards"
    shards.mkdir(parents=True)
    write_tar(shards / "shard-000000.tar", members)
    return MultiTaskDataset(config, model_key=MODEL_KEY, is_train=True, env="local")


def test_transitions_never_rescan_tar_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fraction and resolution steps plus a worker copy read without header scans."""
    dataset = _dataset(tmp_path)
    assert len(dataset.catalog.all_samples) == SAMPLES

    scans: list[Any] = []
    real_scan = tar_offsets.scan_tar_members
    real_open = tarfile.open

    def counting_scan(path: Path):
        scans.append(path)
        return real_scan(path)

    def counting_open(*args: Any, **kwargs: Any):
        scans.append(args[0] if args else kwargs.get("name"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(tar_offsets, "scan_tar_members", counting_scan)
    monkeypatch.setattr(tarfile, "open", counting_open)

    seen: list[set[Any]] = []
    for size, fraction in ((None, 0.25), (32, 0.5), (48, 0.75), (None, 1.0)):
        dataset.update_strategy(size=size, fraction=fraction)
        seen.append(set(dataset.catalog.samples))
        assert dataset[0] is not None
        # A rebuilt loader's spawned workers receive the readers pickled.
        for reader in dataset.catalog.container_readers.values():
            worker_reader = pickle.loads(pickle.dumps(reader))
            assert worker_reader[len(worker_reader) - 1].image_bytes
    assert not scans, f"curriculum transitions rescanned tar headers: {scans}"
    assert [len(s) for s in seen] == [5, 10, 15, 20]
    for smaller, larger in itertools.pairwise(seen):
        assert smaller <= larger, "a larger fraction dropped a sample (ratchet)"
