"""Manifold discovery loads each real folder once (alias names never double the samples).

On Kaggle the data cell links alias names (``UpnV2``, ``LemGendizedUpnV2``) to one attached
folder, and the ``...KaggleReady`` name falls back to that same attached manifold. Before
the fix every name opened its own reader, so each sample was in the epoch twice.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.dataset import MultiTaskDataset
from data.dataset_sources import DatasetSourcesMixin
from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar

MODEL_KEY = "nafnet_denoising"
SAMPLES = 6


def _manifold(root: Path) -> Path:
    """A one-shard WebDataset manifold of :data:`SAMPLES` restoration samples."""
    members: dict[str, bytes] = {}
    for i in range(SAMPLES):
        key = f"s{i:03d}"
        members[f"{key}.webp"] = webp_bytes((i * 10, 90, 160), size=(16, 16), lossless=True)
        members[f"{key}.target.webp"] = webp_bytes((200, 210, i * 10), size=(16, 16), lossless=True)
        members[f"{key}.json"] = json_bytes({"task": "restoration"})
    shards = root / "shards" / "train"
    shards.mkdir(parents=True)
    write_tar(shards / "shard-000000.tar", members)
    return root


def _dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paths: dict[str, str]
) -> MultiTaskDataset:
    """Dataset whose registry names resolve to ``paths`` (name -> folder)."""
    registry_path = Path(__file__).resolve().parents[2] / "unified_models_v2.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert registry[MODEL_KEY]
    monkeypatch.setattr(DatasetSourcesMixin, "_raw_dataset_names", lambda _self: list(paths))
    monkeypatch.setattr(
        DatasetSourcesMixin, "get_dataset_path", lambda _self, name: paths.get(name)
    )
    config: dict[str, Any] = {
        "unified_models": "unified_models_v2.yaml",
        "paths": {"datasets_root": str(tmp_path / "datasets")},
    }
    return MultiTaskDataset(config, model_key=MODEL_KEY, is_train=True, env="local")


def test_symlink_aliases_load_the_manifold_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names linked to one attached folder (the Kaggle alias links) give one reader."""
    real = _manifold(tmp_path / "input" / "LemGendizedUpnV2")
    links = tmp_path / "working"
    links.mkdir()
    for alias in ("UpnV2", "LemGendizedUpnV2"):
        os.symlink(real, links / alias, target_is_directory=True)
    paths = {
        "LemGendizedUpnV2": str(links / "LemGendizedUpnV2"),
        "UpnV2": str(links / "UpnV2"),
        "LemGendizedUpnV2KaggleReady": str(real),
    }
    dataset = _dataset(tmp_path, monkeypatch, paths)
    assert len(dataset.catalog.container_readers) == 1
    assert len(dataset.catalog.all_samples) == SAMPLES
    assert len(set(dataset.catalog.all_samples)) == SAMPLES


def test_unnormalised_paths_of_one_folder_load_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two spellings of one folder (the fuzzy fallback's path) count its samples once."""
    real = _manifold(tmp_path / "datasets" / "LemGendizedUpnV2")
    (tmp_path / "datasets" / "elsewhere").mkdir()
    other = tmp_path / "datasets" / "elsewhere" / ".." / "LemGendizedUpnV2"
    paths = {"LemGendizedUpnV2": str(real), "LemGendizedUpnV2KaggleReady": str(other)}
    dataset = _dataset(tmp_path, monkeypatch, paths)
    assert list(dataset.catalog.container_readers) == ["LemGendizedUpnV2"]
    assert len(dataset.catalog.all_samples) == SAMPLES


def test_distinct_manifolds_are_all_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Different real folders still each contribute their samples."""
    first = _manifold(tmp_path / "datasets" / "LemGendizedA")
    second = _manifold(tmp_path / "datasets" / "LemGendizedB")
    dataset = _dataset(tmp_path, monkeypatch, {"A": str(first), "B": str(second)})
    assert len(dataset.catalog.container_readers) == 2
    assert len(dataset.catalog.all_samples) == 2 * SAMPLES
