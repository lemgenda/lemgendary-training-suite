"""WebDataset container reader for sharded tar archives.

Uncompressed shards are read by byte offset: the member tables come from
:mod:`training.data.containers.tar_offsets` (cached across runs) and every payload is one
``seek`` plus one ``read`` on a binary handle from a bounded per-worker
:class:`~training.data.containers.handle_pool.HandlePool`. Compressed shards keep the
``tarfile`` path, with their handles bounded by the same kind of pool.
"""

from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from training.data.containers.base import Sample
from training.data.containers.handle_pool import DEFAULT_MAX_OPEN_HANDLES, HandlePool
from training.data.containers.tar_offsets import MemberRef, load_member_tables

logger = logging.getLogger("lemtrain.containers.webdataset")

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}

# Index entry modes: offset reads, tarfile reads (compressed shards), loose files.
MODE_OFFSET = "tar"
MODE_TARFILE = "tarfile"
MODE_FILE = "file"


def _parse_member_name(name: str) -> tuple[str, str, str]:
    """Parse tar member name into (sample_key, role, extension).

    Roles: 'image', 'target', 'mask', 'text', 'json', 'other'.
    """
    p = Path(name)
    parts = p.name.split(".")
    ext = p.suffix.lower()

    # Directory-in-tar pattern (e.g. targets/train/sample01.webp)
    parent_parts = {part.lower() for part in p.parent.parts}
    for directory_role in ("target", "mask", "image"):
        if f"{directory_role}s" in parent_parts:
            return p.stem, directory_role, ext

    # WebDataset dot-separated pattern:
    # 000001.target.webp -> key='000001', role='target', ext='.webp'
    # 000001.mask.webp   -> key='000001', role='mask', ext='.webp'
    # 000001.webp        -> key='000001', role='image', ext='.webp'
    if len(parts) >= 3:
        key = parts[0]
        sub = parts[1].lower()
        role = sub
        if sub in ("target", "tgt", "gt", "hr"):
            role = "target"
        elif sub in ("mask", "msk", "seg"):
            role = "mask"
        return key, role, ext

    if len(parts) == 2:
        key = parts[0]
        role = "other"
        if ext in IMAGE_EXTENSIONS:
            role = "image"
        elif ext in (".txt", ".caption"):
            role = "text"
        elif ext == ".json":
            role = "json"
        return key, role, ext

    return p.stem, "image", ext


_IndexEntry = tuple[str, Path | None, Any, Any, Any, Any, Any]


def _find_tar_files(root: Path, cand_dirs: list[Path]) -> list[Path]:
    """Return ``root`` itself, the first candidate dir's shards, or every shard below ``root``.

    ``*.data`` shards win over ``*.tar`` shards in the same place.
    """
    if root.is_file() and root.suffix in (".data", ".tar"):
        return [root]
    for d in cand_dirs:
        if d.exists() and d.is_dir():
            shards = sorted(d.glob("*.data"))
            if not shards:
                shards = sorted(d.glob("*.tar"))
            if shards:
                return shards
    shards = sorted(root.glob("**/*.data"))
    if not shards:
        shards = sorted(root.glob("**/*.tar"))
    return shards


def _sample_entries(
    mode: str, tar_path: Path | None, samples: dict[str, dict[str, Any]]
) -> list[_IndexEntry]:
    """Return one index entry per sample key that has an image role."""
    entries: list[_IndexEntry] = []
    for roles in samples.values():
        img_ref = roles.get("image")
        if img_ref is not None:
            entries.append(
                (
                    mode,
                    tar_path,
                    img_ref,
                    roles.get("target"),
                    roles.get("mask"),
                    roles.get("text"),
                    roles.get("json"),
                )
            )
    return entries


def _offset_entries(t_path: Path, members: list[MemberRef]) -> list[_IndexEntry]:
    """Group an uncompressed shard's members by sample key and role (member offsets).

    A later member with the same key and role replaces an earlier one, as the
    name-keyed ``tarfile`` index did (``extractfile`` returns the last match).
    """
    samples: dict[str, dict[str, MemberRef]] = {}
    for ref in members:
        key, role, _ = _parse_member_name(ref.name)
        samples.setdefault(key, {})[role] = ref
    return _sample_entries(MODE_OFFSET, t_path, samples)


def _index_tar(t_path: Path) -> list[_IndexEntry]:
    """Group a compressed shard's file members by sample key and role (member names)."""
    with tarfile.open(t_path, mode="r:*") as tf:
        members = tf.getmembers()
    samples: dict[str, dict[str, str]] = {}
    for m in members:
        if not m.isfile():
            continue
        key, role, _ = _parse_member_name(m.name)
        samples.setdefault(key, {})[role] = m.name
    return _sample_entries(MODE_TARFILE, t_path, samples)


def _scan_dir_samples(scan_dir: Path) -> dict[str, dict[str, Path]]:
    """Group the image/caption/json files of ``scan_dir`` by sample key and role."""
    samples_dir: dict[str, dict[str, Path]] = {}
    for entry in sorted(scan_dir.iterdir()):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in IMAGE_EXTENSIONS or ext in (".txt", ".caption", ".json"):
            key, role, _ = _parse_member_name(entry.name)
            samples_dir.setdefault(key, {})[role] = entry
    return samples_dir


def _index_shard_dirs(cand_dirs: list[Path]) -> list[_IndexEntry]:
    """Index the first candidate dir whose sub-dirs (or itself) hold samples."""
    index_map: list[_IndexEntry] = []
    for d in cand_dirs:
        if not d.exists() or not d.is_dir():
            continue

        sub_dirs = sorted([sub for sub in d.iterdir() if sub.is_dir()])
        target_dirs = sub_dirs if sub_dirs else [d]

        for scan_dir in target_dirs:
            try:
                samples_dir = _scan_dir_samples(scan_dir)
            except OSError as scan_err:
                logger.debug("Failed scanning directory %s: %s", scan_dir, scan_err)
                continue
            index_map.extend(_sample_entries(MODE_FILE, None, samples_dir))

        if index_map:
            break
    return index_map


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Value of the first key of ``keys`` present in ``data`` with a non-``None`` value.

    Falsy labels such as class ``0`` or score ``0.0`` count as present.
    """
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _json_label(parsed_json: Any, label: Any) -> Any:
    """Fill a missing ``label`` from a JSON sidecar (NIMA scores first, then generic keys).

    A text sidecar label, when present, takes precedence over the JSON one.
    """
    if label is not None or not isinstance(parsed_json, dict):
        return label
    nima_data = parsed_json.get("nima_scores")
    if isinstance(nima_data, dict):
        label = _first_present(nima_data, ("distribution", "scores", "mean_score"))
    if label is None:
        label = _first_present(parsed_json, ("distribution", "scores", "label", "score"))
    return label


def _make_sample(
    image_name: Path,
    payload: tuple[bytes, bytes | None, bytes | None],
    label: Any,
    meta: dict[str, Any],
) -> Sample:
    """``Sample`` named after the image member; ``payload`` is (image, target, mask) bytes."""
    image_bytes, target_bytes, mask_bytes = payload
    return Sample(
        name=image_name.stem.split(".")[0],
        image_format=image_name.suffix.lstrip(".").lower(),
        image_bytes=image_bytes,
        target_bytes=target_bytes,
        mask_bytes=mask_bytes,
        label=label,
        metadata=meta,
    )


def _existing_bytes(path: Path | None) -> bytes | None:
    """The file's bytes, or ``None`` when ``path`` is unset or missing."""
    return path.read_bytes() if path is not None and path.exists() else None


def _read_offset(handle: IO[bytes], ref: MemberRef) -> bytes:
    """Payload of ``ref``; a short read means the shard changed after it was indexed."""
    handle.seek(ref.offset)
    data = handle.read(ref.size)
    if len(data) != ref.size:
        raise OSError(
            f"Short read of tar member {ref.name}: {len(data)} of {ref.size} bytes "
            "(shard changed since it was indexed)."
        )
    return data


def _read_tar_member(tf: tarfile.TarFile, name: str) -> bytes | None:
    """The tar member's bytes, or ``None`` when it is not a regular file."""
    extracted = tf.extractfile(name)
    return extracted.read() if extracted is not None else None


def _open_raw(path: Path) -> IO[bytes]:
    """Binary handle for offset reads (``HandlePool`` opener, picklable by reference)."""
    return open(path, "rb")


def _open_tar(path: Path) -> tarfile.TarFile:
    """``tarfile`` handle for compressed shards (``HandlePool`` opener)."""
    return tarfile.open(path, mode="r:*")


MemberReader = Callable[[Any], "bytes | None"]
"""Reads one member reference of an index entry (a ``None`` reference reads ``None``)."""


def _member_name(ref: Any) -> str:
    """Member name of a reference (``MemberRef`` or plain name)."""
    return ref.name if isinstance(ref, MemberRef) else str(ref)


def _file_sample(entry: tuple) -> Sample:
    """Read one extracted-directory sample (image plus target, mask, text and JSON files)."""
    _, _, img_path, tgt_path, mask_path, txt_path, json_path = entry
    image_bytes = img_path.read_bytes()
    payload = (image_bytes, _existing_bytes(tgt_path), _existing_bytes(mask_path))

    label: Any = None
    meta: dict[str, Any] = {"file": str(img_path)}

    if txt_path is not None and txt_path.exists():
        try:
            label = txt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Unreadable text label %s; sample has no text label: %s", txt_path, exc)

    if json_path is not None and json_path.exists():
        try:
            parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
            meta["json"] = parsed_json
            label = _json_label(parsed_json, label)
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable JSON sidecar %s; metadata skipped: %s", json_path, exc)

    return _make_sample(img_path, payload, label, meta)


def _tar_sidecar_label(
    read: MemberReader, t_path: Path, sidecars: tuple[Any, Any], meta: dict
) -> Any:
    """Label from the text and JSON sidecar members (``sidecars``); JSON goes into ``meta``."""
    txt_m, json_m = sidecars
    label = None

    if txt_m is not None:
        try:
            txt_bytes = read(txt_m)
            if txt_bytes is not None:
                label = txt_bytes.decode("utf-8").strip()
        except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
            logger.warning(
                "Unreadable text label %s in %s; sample has no text label: %s",
                _member_name(txt_m),
                t_path,
                exc,
            )

    if json_m is not None:
        try:
            json_bytes = read(json_m)
            if json_bytes is not None:
                parsed_json = json.loads(json_bytes.decode("utf-8"))
                meta["json"] = parsed_json
                label = _json_label(parsed_json, label)
        except (OSError, tarfile.TarError, ValueError) as exc:
            logger.warning(
                "Unreadable JSON sidecar %s in %s; metadata skipped: %s",
                _member_name(json_m),
                t_path,
                exc,
            )
    return label


class WebDatasetReader:
    """Reader for tar-sharded WebDataset archives with support for unpacked shard directories.

    ``max_open_handles`` caps the shard handles each process (dataloader worker) keeps open.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        max_open_handles: int = DEFAULT_MAX_OPEN_HANDLES,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = split

        cand_dirs = [
            self.root / "shards" / self.split,
            self.root / self.split / "shards",
            self.root / "shards",
            self.root / self.split,
            self.root,
        ]
        self.tar_files: list[Path] = _find_tar_files(self.root, cand_dirs)

        # Map sample index -> (mode, tar_path, img_ref, tgt_ref, mask_ref, txt_ref, json_ref)
        self.index_map: list[_IndexEntry] = []
        self._raw_handles: HandlePool[IO[bytes]] = HandlePool(_open_raw, max_open_handles)
        self._tar_handles: HandlePool[tarfile.TarFile] = HandlePool(_open_tar, max_open_handles)

        if self.tar_files:
            self._index_shards()
        else:
            # Mode B: Extracted shard directories or loose image files
            self.index_map.extend(_index_shard_dirs(cand_dirs))

        if not self.index_map and self.split == "train":
            raise FileNotFoundError(
                f"No WebDataset tar archives or extracted shard directories found in '{self.root}' "
                f"for split '{self.split}'."
            )

    def _index_shards(self) -> None:
        """Index every shard: cached or scanned offsets, else names for compressed shards."""
        tables = load_member_tables(self.tar_files)
        for t_path in self.tar_files:
            if t_path not in tables:
                continue
            table = tables[t_path]
            if table is not None:
                self.index_map.extend(_offset_entries(t_path, table))
                continue
            try:
                self.index_map.extend(_index_tar(t_path))
            except (OSError, tarfile.TarError, EOFError) as exc:
                logger.warning("Failed opening tar file %s: %s", t_path, exc)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.index_map):
            raise IndexError(
                f"WebDataset index {index} out of range (0..{len(self.index_map) - 1})."
            )

        entry = self.index_map[index]
        if entry[0] == MODE_FILE:
            return _file_sample(entry)
        return self._tar_sample(entry, index)

    def shard_of(self, index: int) -> Path | None:
        """Shard file holding sample ``index`` (``None`` for loose-file samples)."""
        return self.index_map[index][1]

    def open_handle_count(self) -> int:
        """Shard handles this process currently keeps open."""
        return len(self._raw_handles) + len(self._tar_handles)

    def _member_reader(self, mode: str, t_path: Path) -> MemberReader:
        """Reader of one entry's member references for its shard and mode."""
        if mode == MODE_OFFSET:
            handle = self._raw_handles.get(t_path)
            return lambda ref: None if ref is None else _read_offset(handle, ref)
        tf = self._tar_handles.get(t_path)
        return lambda ref: None if ref is None else _read_tar_member(tf, ref)

    def _tar_sample(self, entry: tuple, index: int) -> Sample:
        """Read one sample's members (image, target, mask, text, JSON) from its tar archive."""
        t_path = entry[1]
        if t_path is None or not isinstance(t_path, Path):
            raise ValueError(
                f"Corrupted WebDataset index entry: tar archive path is None at index {index}."
            )

        read = self._member_reader(entry[0], t_path)
        payload = self._read_payload(read, t_path, entry[2:5])
        img_name = _member_name(entry[2])
        meta: dict[str, Any] = {"tar_file": str(t_path), "member": img_name}
        label = _tar_sidecar_label(read, t_path, (entry[5], entry[6]), meta)
        return _make_sample(Path(img_name), payload, label, meta)

    def _read_payload(
        self, read: MemberReader, t_path: Path, refs: tuple[Any, Any, Any]
    ) -> tuple[bytes, bytes | None, bytes | None]:
        """Image, target and mask bytes; a failed read drops the shard's handle and raises."""
        img_m, tgt_m, mask_m = refs
        try:
            return (read(img_m) or b"", read(tgt_m), read(mask_m))
        except OSError:
            # Drop the handle so the next access reopens the shard, then report the failure.
            self._raw_handles.discard(t_path)
            self._tar_handles.discard(t_path)
            raise

    def close(self) -> None:
        """Close all open shard handles of this process."""
        self._raw_handles.close()
        self._tar_handles.close()
