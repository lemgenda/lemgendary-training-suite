"""Byte-offset WebDatasetReader: equivalence with tarfile reads, cache, handle bound, labels."""

from __future__ import annotations

import io
import json
import os
import pickle
import tarfile
from pathlib import Path
from typing import Literal

import pytest

from training.data.containers import tar_offsets
from training.data.containers.handle_pool import HandlePool
from training.data.containers.tar_offsets import (
    CACHE_FILENAME,
    MemberRef,
    ShardOffsetCache,
    scan_tar_members,
)
from training.data.containers.webdataset import MODE_OFFSET, MODE_TARFILE, WebDatasetReader

Members = list[tuple[str, bytes]]


def _published_layout(shard: int, count: int) -> Members:
    """Published manifolds: image, paired target and a JSON label sidecar."""
    members: Members = []
    for i in range(count):
        key = f"s{shard:02d}_{i:04d}"
        members.append((f"{key}.webp", os.urandom(64 + i)))
        members.append((f"{key}.target.webp", os.urandom(32 + i)))
        members.append((f"{key}.json", json.dumps({"label": i % 3}).encode("utf-8")))
    return members


def _compiler_layout(shard: int, count: int) -> Members:
    """Current compiler output: image, lossless mask and a JSON label document."""
    members: Members = []
    for i in range(count):
        key = f"c{shard:02d}_{i:04d}"
        members.append((f"{key}.webp", os.urandom(48 + i)))
        members.append((f"{key}.mask.webp", os.urandom(16 + i)))
        members.append((f"{key}.json", json.dumps({"class_id": i % 5}).encode("utf-8")))
    return members


def _write_shard(path: Path, members: Members, mode: Literal["w", "w:gz"] = "w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _reference_bytes(shard: Path, name: str | None) -> bytes | None:
    """Member bytes read through ``tarfile`` (the pre-offset read path)."""
    if name is None:
        return None
    with tarfile.open(shard, "r:*") as tf:
        extracted = tf.extractfile(name)
        return extracted.read() if extracted is not None else None


def _ref_name(ref: object) -> str | None:
    if ref is None:
        return None
    return ref.name if isinstance(ref, MemberRef) else str(ref)


@pytest.mark.parametrize("layout", [_published_layout, _compiler_layout])
def test_offset_reads_match_tarfile_reads(tmp_path: Path, layout) -> None:
    """Offset reads return the bytes tarfile returns, in both shard layouts."""
    for shard in range(2):
        _write_shard(tmp_path / "shards" / "train" / f"shard-{shard:05d}.data", layout(shard, 6))
    reader = WebDatasetReader(tmp_path, split="train")
    try:
        assert len(reader) == 12
        for index, entry in enumerate(reader.index_map):
            mode, shard_path, img, tgt, mask, _json = entry
            assert mode == MODE_OFFSET and shard_path is not None
            sample = reader[index]
            assert sample.image_bytes == _reference_bytes(shard_path, _ref_name(img))
            assert sample.target_bytes == _reference_bytes(shard_path, _ref_name(tgt))
            assert sample.mask_bytes == _reference_bytes(shard_path, _ref_name(mask))
            assert sample.metadata["member"] == _ref_name(img)
            assert sample.label is not None
    finally:
        reader.close()


def test_compressed_shard_uses_tarfile_reads(tmp_path: Path) -> None:
    """A gzip shard cannot be read by offset and falls back to tarfile reads."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.tar", _compiler_layout(0, 4), "w:gz")
    reader = WebDatasetReader(tmp_path, split="train")
    try:
        assert {entry[0] for entry in reader.index_map} == {MODE_TARFILE}
        for index, entry in enumerate(reader.index_map):
            assert reader[index].image_bytes == _reference_bytes(shard, entry[2])
    finally:
        reader.close()
    assert scan_tar_members(shard) is None


def test_duplicate_member_keeps_last_like_tarfile(tmp_path: Path) -> None:
    """A repeated member name resolves to its last copy, as extractfile does."""
    members = [("a.webp", b"old"), ("a.json", b'{"label": 1}'), ("a.webp", b"new")]
    shard = _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        assert reader[0].image_bytes == b"new" == _reference_bytes(shard, "a.webp")
    finally:
        reader.close()


def test_cache_is_reused_and_invalidated_by_shard_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh cache skips the header scan; a changed shard is rescanned."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 3))
    WebDatasetReader(tmp_path).close()
    assert (shard.parent / CACHE_FILENAME).is_file()

    scans: list[Path] = []
    real_scan = tar_offsets.scan_tar_members

    def counting_scan(path: Path):
        scans.append(path)
        return real_scan(path)

    monkeypatch.setattr(tar_offsets, "scan_tar_members", counting_scan)
    cached = WebDatasetReader(tmp_path)
    cached.close()
    assert not scans, "a fresh cache must not rescan tar headers"

    _write_shard(shard, _published_layout(0, 5))
    stat = shard.stat()
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    rebuilt = WebDatasetReader(tmp_path)
    try:
        assert scans == [shard]
        assert len(rebuilt) == 5
        last = rebuilt.index_map[-1]
        assert rebuilt[4].image_bytes == _reference_bytes(shard, _ref_name(last[2]))
    finally:
        rebuilt.close()


def test_malformed_cache_falls_back_to_scan(tmp_path: Path) -> None:
    """An unreadable cache file is ignored and rebuilt from a scan."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 2))
    (shard.parent / CACHE_FILENAME).write_text("{not json", encoding="utf-8")
    reader = WebDatasetReader(tmp_path)
    try:
        assert len(reader) == 2
    finally:
        reader.close()
    entry = ShardOffsetCache(shard.parent).members(shard)
    assert entry is not None and len(entry) == 6


def test_out_of_range_cache_entry_is_rejected(tmp_path: Path) -> None:
    """A cached member past the shard's end invalidates the entry."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 2))
    cache = ShardOffsetCache(shard.parent)
    cache.store(shard, [MemberRef("x.webp", shard.stat().st_size, 10)])
    cache.save()
    assert ShardOffsetCache(shard.parent).members(shard) is None


def test_read_only_shard_dir_uses_temp_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable shard directory caches in the temp-dir fallback."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 2))
    fallback = tmp_path / "fallback" / "cache.json"
    monkeypatch.setattr(tar_offsets, "_fallback_cache_path", lambda _dir: fallback)
    real_write = Path.write_text

    def refuse_primary(self: Path, *args, **kwargs):
        if self.parent == shard.parent:
            raise PermissionError("read-only shard directory")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refuse_primary)
    cache = ShardOffsetCache(shard.parent)
    cache.store(shard, scan_tar_members(shard) or [])
    assert cache.save() == fallback
    assert ShardOffsetCache(shard.parent).members(shard) is not None


def test_stale_offsets_fail_loudly_and_drop_the_handle(tmp_path: Path) -> None:
    """A shard truncated after indexing raises and drops its handle."""
    shard = _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 3))
    reader = WebDatasetReader(tmp_path)
    try:
        with open(shard, "r+b") as fh:
            fh.truncate(512)
        with pytest.raises(OSError, match="Short read"):
            assert reader[2].image_bytes
        assert reader.open_handle_count() == 0
    finally:
        reader.close()


def test_handle_pool_is_bounded_per_reader(tmp_path: Path) -> None:
    """Reading 20 shards keeps at most 16 handles open."""
    for shard in range(20):
        _write_shard(tmp_path / "shards" / f"shard-{shard:05d}.data", _published_layout(shard, 1))
    reader = WebDatasetReader(tmp_path, max_open_handles=16)
    try:
        for index, _entry in enumerate(reader.index_map):
            assert reader[index].image_bytes
            assert reader.open_handle_count() <= 16
        assert reader.open_handle_count() == 16
    finally:
        reader.close()
    assert reader.open_handle_count() == 0


def test_handle_pool_closes_evicted_handles(tmp_path: Path) -> None:
    """The least recently used handle is closed on eviction."""
    opened: list[io.BytesIO] = []

    def opener(_path: Path) -> io.BytesIO:
        opened.append(io.BytesIO(b"x"))
        return opened[-1]

    pool: HandlePool[io.BytesIO] = HandlePool(opener, max_open=2)
    for name in ("a", "b", "a", "c"):
        pool.get(tmp_path / name)
    assert len(pool) == 2
    # "b" was least recently used when "c" arrived.
    assert [h.closed for h in opened] == [False, True, False]
    with pytest.raises(ValueError):
        HandlePool(opener, max_open=0)


def test_reader_pickles_without_handles(tmp_path: Path) -> None:
    """A pickled reader (worker copy) carries no handles and still reads."""
    _write_shard(tmp_path / "shards" / "shard-00000.data", _published_layout(0, 2))
    reader = WebDatasetReader(tmp_path)
    try:
        expected = reader[1].image_bytes
        clone = pickle.loads(pickle.dumps(reader))
    finally:
        reader.close()
    try:
        assert clone.open_handle_count() == 0
        assert clone[1].image_bytes == expected
    finally:
        clone.close()


@pytest.mark.parametrize(
    ("sidecar", "expected"),
    [
        ({"label": 0}, 0),
        ({"label": 0, "score": 7}, 0),
        ({"nima_scores": {"mean_score": 0.0}}, 0.0),
        ({"score": 0}, 0),
        ({"label": None, "score": 2}, 2),
    ],
)
def test_falsy_json_labels_survive(tmp_path: Path, sidecar: dict, expected: object) -> None:
    """JSON labels 0 and 0.0 are kept instead of falling through."""
    members = [("k.webp", b"img"), ("k.json", json.dumps(sidecar).encode("utf-8"))]
    _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        assert reader[0].label == expected
    finally:
        reader.close()


def test_text_member_is_not_a_label(tmp_path: Path) -> None:
    """Labels come only from the JSON document; a ``.txt`` member is not read."""
    members = [("k.webp", b"img"), ("k.txt", b"3"), ("k.json", b'{"class_id": 0}')]
    _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        sample = reader[0]
        assert sample.label == 0
        assert sample.metadata["json"] == {"class_id": 0}
    finally:
        reader.close()


def test_text_member_without_json_gives_no_label(tmp_path: Path) -> None:
    """A sample with only a ``.txt`` member has no label and no JSON document."""
    members = [("k.webp", b"img"), ("k.txt", b"3")]
    _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        sample = reader[0]
        assert sample.label is None
        assert "json" not in sample.metadata
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("doc", "expected"),
    [({"class_name": "cat"}, "cat"), ({"caption": "a red house"}, "a red house")],
)
def test_schema_label_keys(tmp_path: Path, doc: dict, expected: object) -> None:
    """``class_name`` and ``caption`` documents give their label value."""
    members = [("k.webp", b"img"), ("k.json", json.dumps(doc).encode("utf-8"))]
    _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        assert reader[0].label == expected
    finally:
        reader.close()


def test_malformed_json_is_recorded_not_dropped(tmp_path: Path) -> None:
    """An unreadable JSON member is reported under ``json_error`` (no silent label)."""
    members = [("k.webp", b"img"), ("k.json", b"{not json")]
    _write_shard(tmp_path / "shards" / "shard-00000.tar", members)
    reader = WebDatasetReader(tmp_path)
    try:
        sample = reader[0]
        assert sample.label is None
        assert "json" not in sample.metadata
        assert sample.metadata["json_error"]
    finally:
        reader.close()
