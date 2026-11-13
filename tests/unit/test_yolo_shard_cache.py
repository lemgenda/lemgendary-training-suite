"""YOLOv8n trains from WebDataset shards through a materialised Ultralytics cache (R4-J2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from data import yolo_config_gen, yolo_shard_cache
from data.yolo_shard_cache import (
    DATA_YAML,
    YoloCacheError,
    materialise_yolo_cache,
    split_shards,
)
from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar

CLASSES = ["person", "car", "dog"]
_IMAGE = webp_bytes((10, 20, 30), size=(16, 16), lossless=True)


def _doc(*objects: dict[str, Any]) -> bytes:
    return json_bytes({"object_count": len(objects), "objects": list(objects)})


def _manifold(tmp_path: Path, split_dirs: bool = True, classes: list[str] | None = None) -> Path:
    """Tiny manifold: two train samples (one labelled by name), one background, one val."""
    root = tmp_path / "LemGendizedYoloV8n"
    train = root / "shards" / "train" if split_dirs else root / "shards"
    train.mkdir(parents=True)
    write_tar(
        train / "shard-000000.tar",
        {
            "a.webp": _IMAGE,
            "a.json": _doc(
                {"category_id": 1, "bbox_cxcywh_normalized": [0.5, 0.5, 0.25, 0.5]},
                {"class_name": "dog", "bbox_cxcywh_normalized": [0.1, 0.2, 0.1, 0.1]},
            ),
            "b.webp": _IMAGE,
            "b.json": _doc(),
            "c.webp": _IMAGE,
        },
    )
    if split_dirs:
        val = root / "shards" / "val"
        val.mkdir(parents=True)
        write_tar(
            val / "shard-000000.tar",
            {
                "v.webp": _IMAGE,
                "v.json": _doc({"category_id": 0, "bbox_cxcywh_normalized": [0.3, 0.3, 0.2, 0.2]}),
            },
        )
    info = {"name": root.name, "classes": CLASSES if classes is None else classes}
    (root / "dataset_info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
    return root


def test_materialises_images_labels_and_data_yaml(tmp_path: Path) -> None:
    """Images, YOLO label text and data.yaml are written from the shards."""
    cache = materialise_yolo_cache(_manifold(tmp_path), tmp_path / "cache")
    root = tmp_path / "cache"
    assert sorted(p.name for p in (root / "images" / "train").iterdir()) == [
        "a.webp",
        "b.webp",
        "c.webp",
    ]
    assert (root / "images" / "train" / "a.webp").read_bytes() == _IMAGE
    assert (root / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == (
        "1 0.500000 0.500000 0.250000 0.500000\n2 0.100000 0.200000 0.100000 0.100000\n"
    )
    assert (root / "labels" / "train" / "b.txt").read_text(encoding="utf-8") == ""
    assert not (root / "labels" / "train" / "c.txt").exists()
    assert (root / "labels" / "val" / "v.txt").read_text(encoding="utf-8").startswith("0 ")
    assert cache.images == {"train": 3, "val": 1} and cache.unlabeled == 1
    data = yaml.safe_load((root / DATA_YAML).read_text(encoding="utf-8"))
    assert data == {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": dict(enumerate(CLASSES)),
    }


def test_rerun_skips_finished_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rerun over unchanged shards writes nothing (idempotent)."""
    manifold = _manifold(tmp_path)
    first = materialise_yolo_cache(manifold, tmp_path / "cache")

    def _no_rewrite(*_args: Any) -> None:
        raise AssertionError("an unchanged shard must not be rewritten")

    monkeypatch.setattr(yolo_shard_cache.ShardWriter, "write_shard", _no_rewrite)
    assert materialise_yolo_cache(manifold, tmp_path / "cache") == first


def test_resume_redoes_only_unfinished_or_changed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an interrupted or changed shard is redone; stale samples go."""
    manifold = _manifold(tmp_path)
    cache_dir = tmp_path / "cache"
    materialise_yolo_cache(manifold, cache_dir)
    (cache_dir / ".progress" / "val" / "shard-000000.tar.json").unlink()
    train_shard = manifold / "shards" / "train" / "shard-000000.tar"
    write_tar(train_shard, {"d.webp": _IMAGE, "d.json": _doc()})

    written: list[str] = []
    real = yolo_shard_cache.ShardWriter.write_shard

    def _counting(writer: Any, shard: Path) -> Any:
        written.append(f"{shard.parent.name}/{shard.name}")
        return real(writer, shard)

    monkeypatch.setattr(yolo_shard_cache.ShardWriter, "write_shard", _counting)
    cache = materialise_yolo_cache(manifold, cache_dir)
    assert sorted(written) == ["train/shard-000000.tar", "val/shard-000000.tar"]
    assert cache.images == {"train": 1, "val": 1}
    assert sorted(p.name for p in (cache_dir / "images" / "train").iterdir()) == ["d.webp"]
    assert sorted(p.name for p in (cache_dir / "labels" / "train").iterdir()) == ["d.txt"]


def test_flat_shards_validate_on_train(tmp_path: Path) -> None:
    """A flat shards folder is the train split; val reuses it."""
    manifold = _manifold(tmp_path, split_dirs=False)
    assert list(split_shards(manifold)) == ["train"]
    cache = materialise_yolo_cache(manifold, tmp_path / "cache")
    assert cache.config()["val"] == "images/train"


def test_missing_classes_raise(tmp_path: Path) -> None:
    """Without dataset_info.yaml classes there is no data.yaml."""
    manifold = _manifold(tmp_path, classes=[])
    with pytest.raises(YoloCacheError, match="no classes list"):
        materialise_yolo_cache(manifold, tmp_path / "cache")


@pytest.mark.parametrize(
    ("label", "match"),
    [
        (b"{oops", "unreadable JSON label"),
        (json_bytes({"caption": "no objects"}), "no objects list"),
        (
            _doc({"category_id": 7, "bbox_cxcywh_normalized": [0.5, 0.5, 0.1, 0.1]}),
            "outside the 3 dataset_info.yaml classes",
        ),
        (
            _doc({"class_name": "cat", "bbox_cxcywh_normalized": [0.5, 0.5, 0.1, 0.1]}),
            "not a manifold class",
        ),
    ],
)
def test_bad_labels_raise(tmp_path: Path, label: bytes, match: str) -> None:
    """Unreadable, object-less or out-of-range labels stop the cache build."""
    manifold = _manifold(tmp_path)
    write_tar(manifold / "shards" / "val" / "shard-000000.tar", {"v.webp": _IMAGE, "v.json": label})
    with pytest.raises(YoloCacheError, match=match):
        materialise_yolo_cache(manifold, tmp_path / "cache")


def test_generate_yolo_yaml_uses_the_shard_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The YOLO training path gets the cache's data.yaml for a sharded manifold."""
    manifold = _manifold(tmp_path)
    monkeypatch.setattr(
        yolo_config_gen.ManifoldResolver, "resolve_manifold", lambda _self, _name: manifold
    )
    monkeypatch.setattr(
        yolo_config_gen, "yolo_cache_dir", lambda _root, name: tmp_path / "cache" / name
    )
    registry = {"yolov8n": {"datasets": [manifold.name]}}
    path = yolo_config_gen.generate_yolo_yaml({"env": "local"}, "yolov8n", registry)
    assert path is not None
    assert Path(path) == tmp_path / "cache" / manifold.name / DATA_YAML
    assert yaml.safe_load(Path(path).read_text(encoding="utf-8"))["nc"] == len(CLASSES)
