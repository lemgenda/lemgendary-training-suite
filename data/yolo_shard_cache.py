"""Materialise an Ultralytics images/labels dataset from a manifold's WebDataset shards.

Ultralytics trains from image files with one YOLO ``.txt`` label each
(``images/<split>/<key>.<ext>``, ``labels/<split>/<key>.txt``) and a ``data.yaml``. The
compiled manifolds hold WebDataset shards whose labels are JSON documents, so YOLOv8n
reads them through this cache:

* every image member is written to ``<cache>/images/<split>/``,
* its JSON ``objects`` become ``<cache>/labels/<split>/<key>.txt`` (``cls cx cy w h``;
  an object's ``class_name`` maps to its id through ``dataset_info.yaml`` ``classes``),
* ``<cache>/data.yaml`` names the classes from ``dataset_info.yaml`` ``classes``.

Shards are ``shards/<split>/*.tar|*.data`` (a flat ``shards/`` folder is the train
split); without val shards, val reuses the train images. A sample without a JSON label
is a background image (YOLO convention: no label file) and is counted; an unreadable or
malformed label raises :class:`YoloCacheError`.

The cache is idempotent and resumable: each finished shard leaves a progress record
(its size and modification time), so a rerun skips unchanged shards and redoes only a
shard that was interrupted or has changed since (a changed shard's earlier images and
labels are removed first, so no stale sample stays in the cache).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from data.json_labels import LabelDocumentError, YoloRow, sample_document, yolo_rows
from data.manifold_classes import manifold_classes
from training.data.containers.base import Sample
from training.data.containers.webdataset import WebDatasetReader

logger = logging.getLogger("lemtrain.data.yolo_shard_cache")

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
SPLITS = (TRAIN_SPLIT, VAL_SPLIT)
SHARD_SUFFIXES = (".tar", ".data")
DATA_YAML = "data.yaml"
PROGRESS_DIR = ".progress"


class YoloCacheError(RuntimeError):
    """The shards cannot become an Ultralytics dataset (no classes, a bad label ...)."""


@dataclass(frozen=True)
class ShardRecord:
    """A materialised shard: the shard state it was built from and what it held."""

    size: int
    mtime_ns: int
    unlabeled: int
    files: tuple[str, ...]

    @property
    def images(self) -> int:
        """Images the shard held."""
        return len(self.files)

    def matches(self, shard: Path) -> bool:
        """True when ``shard`` is unchanged since this record was written."""
        stat = shard.stat()
        return stat.st_size == self.size and stat.st_mtime_ns == self.mtime_ns


@dataclass(frozen=True)
class YoloCache:
    """A materialised Ultralytics dataset."""

    root: Path
    names: tuple[str, ...]
    images: dict[str, int]
    unlabeled: int
    val_split: str

    @property
    def data_yaml(self) -> Path:
        """The Ultralytics ``data.yaml`` of the cache."""
        return self.root / DATA_YAML

    def config(self) -> dict[str, Any]:
        """The ``data.yaml`` content."""
        return {
            "path": str(self.root),
            "train": f"images/{TRAIN_SPLIT}",
            "val": f"images/{self.val_split}",
            "nc": len(self.names),
            "names": dict(enumerate(self.names)),
        }


def _shards_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in SHARD_SUFFIXES)


def split_shards(manifold_dir: Path) -> dict[str, list[Path]]:
    """Shards per split: ``shards/<split>/``, else a flat ``shards/`` folder as train."""
    shards_dir = manifold_dir / "shards"
    found = {split: _shards_in(shards_dir / split) for split in SPLITS}
    if not found[TRAIN_SPLIT]:
        found[TRAIN_SPLIT] = _shards_in(shards_dir)
    return {split: shards for split, shards in found.items() if shards}


def has_shards(manifold_dir: Path) -> bool:
    """True when the manifold holds WebDataset shards this cache can read."""
    return TRAIN_SPLIT in split_shards(manifold_dir)


def label_lines(rows: list[YoloRow], class_count: int) -> str:
    """YOLO label text of ``rows``; :class:`LabelDocumentError` for a class out of range."""
    lines = []
    for cls, cx, cy, w, h in rows:
        if cls >= class_count:
            raise LabelDocumentError(
                f"class id {cls} is outside the {class_count} dataset_info.yaml classes"
            )
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "".join(f"{line}\n" for line in lines)


class ShardWriter:
    """Writes one split's images and labels into the cache."""

    def __init__(self, root: Path, split: str, classes: tuple[str, ...]) -> None:
        self.images_dir = root / "images" / split
        self.labels_dir = root / "labels" / split
        self.progress_dir = root / PROGRESS_DIR / split
        self.classes = classes
        for folder in (self.images_dir, self.labels_dir, self.progress_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def _record_path(self, shard: Path) -> Path:
        return self.progress_dir / f"{shard.name}.json"

    def record(self, shard: Path) -> ShardRecord | None:
        """The progress record of ``shard`` (None when there is none or it is unreadable)."""
        path = self._record_path(shard)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ShardRecord(**{**raw, "files": tuple(raw["files"])})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Ignoring unreadable progress record %s: %s", path, exc)
            return None

    def remove(self, record: ShardRecord) -> None:
        """Delete the images and labels an earlier version of a shard wrote."""
        for name in record.files:
            (self.images_dir / name).unlink(missing_ok=True)
            (self.labels_dir / f"{Path(name).stem}.txt").unlink(missing_ok=True)

    def write_shard(self, shard: Path) -> ShardRecord:
        """Write every sample of ``shard``; record the shard as finished last."""
        stat = shard.stat()
        reader = WebDatasetReader(shard, split=TRAIN_SPLIT)
        files: list[str] = []
        unlabeled = 0
        try:
            for sample in (reader[index] for index in range(len(reader))):
                unlabeled += self._write_sample(sample, shard)
                files.append(_image_name(sample))
        finally:
            reader.close()
        record = ShardRecord(stat.st_size, stat.st_mtime_ns, unlabeled, tuple(files))
        _write_atomic(self._record_path(shard), json.dumps(asdict(record)))
        return record

    def _write_sample(self, sample: Sample, shard: Path) -> int:
        """Write one image and its label; 1 when it has no JSON label (background)."""
        where = f"sample '{sample.name}' in {shard}"
        try:
            doc = sample_document(sample.metadata, where)
            rows = yolo_rows(doc, self.classes) if doc is not None else None
            if doc is not None and rows is None:
                raise LabelDocumentError("the JSON label has no objects list")
            text = label_lines(rows, len(self.classes)) if rows is not None else None
        except LabelDocumentError as exc:
            raise YoloCacheError(f"{where}: {exc}") from exc
        (self.images_dir / _image_name(sample)).write_bytes(sample.image_bytes)
        label_path = self.labels_dir / f"{sample.name}.txt"
        if text is None:
            label_path.unlink(missing_ok=True)
            return 1
        label_path.write_text(text, encoding="utf-8")
        return 0


def _image_name(sample: Sample) -> str:
    return f"{sample.name}.{sample.image_format}"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _materialise_split(writer: ShardWriter, shards: list[Path]) -> tuple[int, int]:
    """Images and unlabeled images of one split, skipping finished shards."""
    images = unlabeled = 0
    for shard in shards:
        record = writer.record(shard)
        if record is None or not record.matches(shard):
            if record is not None:
                writer.remove(record)
            logger.info("Materialising YOLO samples of %s", shard)
            record = writer.write_shard(shard)
        images += record.images
        unlabeled += record.unlabeled
    return images, unlabeled


def materialise_yolo_cache(manifold_dir: Path, cache_dir: Path) -> YoloCache:
    """Build (or resume) the Ultralytics dataset of ``manifold_dir`` under ``cache_dir``.

    Raises :class:`YoloCacheError` when the manifold has no train shards, no
    ``dataset_info.yaml`` ``classes``, or a sample with an unreadable or malformed label.
    """
    shards = split_shards(manifold_dir)
    if TRAIN_SPLIT not in shards:
        raise YoloCacheError(f"{manifold_dir} has no WebDataset shards under shards/")
    names = manifold_classes(manifold_dir)
    if not names:
        raise YoloCacheError(
            f"{manifold_dir / 'dataset_info.yaml'} has no classes list; the YOLO data.yaml "
            "needs the class names in id order."
        )
    counts: dict[str, int] = {}
    unlabeled = 0
    for split, split_list in shards.items():
        counts[split], split_unlabeled = _materialise_split(
            ShardWriter(cache_dir, split, names), split_list
        )
        unlabeled += split_unlabeled
    val_split = VAL_SPLIT if VAL_SPLIT in shards else TRAIN_SPLIT
    if val_split != VAL_SPLIT:
        logger.warning("%s has no val shards; YOLO validates on the train images.", manifold_dir)
    if unlabeled:
        logger.warning(
            "%s: %d images without a JSON label train as background images.",
            manifold_dir,
            unlabeled,
        )
    cache = YoloCache(cache_dir, names, counts, unlabeled, val_split)
    _write_atomic(cache.data_yaml, yaml.safe_dump(cache.config(), sort_keys=False))
    return cache
