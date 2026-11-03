"""WebDataset container reader for sharded tar archives."""

from __future__ import annotations

import json
import logging
import tarfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.webdataset")

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}


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


def _index_tar(t_path: Path) -> list[_IndexEntry]:
    """Group a tar shard's file members by sample key and role (member names)."""
    with tarfile.open(t_path, mode="r:*") as tf:
        members = tf.getmembers()
    samples: dict[str, dict[str, str]] = {}
    for m in members:
        if not m.isfile():
            continue
        key, role, _ = _parse_member_name(m.name)
        samples.setdefault(key, {})[role] = m.name
    return _sample_entries("tar", t_path, samples)


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
            index_map.extend(_sample_entries("file", None, samples_dir))

        if index_map:
            break
    return index_map


class WebDatasetReader:
    """Reader for tar-sharded WebDataset archives with support for unpacked shard directories."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
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
        self.index_map: list[tuple[str, Path | None, Any, Any, Any, Any, Any]] = []
        self._tar_handles: dict[Path, tarfile.TarFile] = {}
        # Owns the cached tar handles; close() unwinds it.
        self._tar_stack = ExitStack()

        if self.tar_files:
            for t_path in self.tar_files:
                try:
                    self.index_map.extend(_index_tar(t_path))
                except (OSError, tarfile.TarError, EOFError) as exc:
                    logger.warning("Failed opening tar file %s: %s", t_path, exc)
        else:
            # Mode B: Extracted shard directories or loose image files
            self.index_map.extend(_index_shard_dirs(cand_dirs))

        if not self.index_map and self.split == "train":
            raise FileNotFoundError(
                f"No WebDataset tar archives or extracted shard directories found in '{self.root}' "
                f"for split '{self.split}'."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.index_map):
            raise IndexError(
                f"WebDataset index {index} out of range (0..{len(self.index_map) - 1})."
            )

        entry = self.index_map[index]
        mode = entry[0]

        if mode == "file":
            _, _, img_path, tgt_path, mask_path, txt_path, json_path = entry
            image_bytes = img_path.read_bytes()
            stem = img_path.stem.split(".")[0]
            fmt = img_path.suffix.lstrip(".").lower()

            target_bytes = (
                tgt_path.read_bytes() if tgt_path is not None and tgt_path.exists() else None
            )
            mask_bytes = (
                mask_path.read_bytes() if mask_path is not None and mask_path.exists() else None
            )

            label: Any = None
            meta: dict[str, Any] = {"file": str(img_path)}

            if txt_path is not None and txt_path.exists():
                try:
                    label = txt_path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeDecodeError) as exc:
                    logger.warning(
                        "Unreadable text label %s; sample has no text label: %s", txt_path, exc
                    )

            if json_path is not None and json_path.exists():
                try:
                    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
                    meta["json"] = parsed_json
                    if label is None and isinstance(parsed_json, dict):
                        nima_data = parsed_json.get("nima_scores")
                        if isinstance(nima_data, dict):
                            label = (
                                nima_data.get("distribution")
                                or nima_data.get("scores")
                                or nima_data.get("mean_score")
                            )
                        if label is None:
                            label = (
                                parsed_json.get("distribution")
                                or parsed_json.get("scores")
                                or parsed_json.get("label")
                                or parsed_json.get("score")
                            )
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Unreadable JSON sidecar %s; metadata skipped: %s", json_path, exc
                    )

            return Sample(
                name=stem,
                image_bytes=image_bytes,
                image_format=fmt,
                target_bytes=target_bytes,
                mask_bytes=mask_bytes,
                label=label,
                metadata=meta,
            )

        # Mode: tar
        _, t_path, img_m, tgt_m, mask_m, txt_m, json_m = entry
        if t_path is None or not isinstance(t_path, Path):
            raise ValueError(
                f"Corrupted WebDataset index entry: tar archive path is None at index {index}."
            )
        assert isinstance(t_path, Path)

        tf = self._tar_handles.get(t_path)
        if tf is None:
            # Open under a temporary stack, then hand ownership to the long-lived one.
            with ExitStack() as stack:
                tf = stack.enter_context(tarfile.open(t_path, mode="r:*"))
                self._tar_stack.push(stack.pop_all())
            self._tar_handles[t_path] = tf

        extracted = tf.extractfile(img_m)
        image_bytes = extracted.read() if extracted is not None else b""
        stem = Path(img_m).stem.split(".")[0]
        fmt = Path(img_m).suffix.lstrip(".").lower()

        target_bytes: bytes | None = None
        if tgt_m is not None:
            tgt_extracted = tf.extractfile(tgt_m)
            if tgt_extracted is not None:
                target_bytes = tgt_extracted.read()

        mask_bytes: bytes | None = None
        if mask_m is not None:
            mask_extracted = tf.extractfile(mask_m)
            if mask_extracted is not None:
                mask_bytes = mask_extracted.read()

        label = None
        meta = {"tar_file": str(t_path), "member": img_m}

        if txt_m is not None:
            txt_extracted = tf.extractfile(txt_m)
            if txt_extracted is not None:
                try:
                    label = txt_extracted.read().decode("utf-8").strip()
                except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
                    logger.warning(
                        "Unreadable text label %s in %s; sample has no text label: %s",
                        txt_m,
                        t_path,
                        exc,
                    )

        if json_m is not None:
            json_extracted = tf.extractfile(json_m)
            if json_extracted is not None:
                try:
                    parsed_json = json.loads(json_extracted.read().decode("utf-8"))
                    meta["json"] = parsed_json
                    if label is None and isinstance(parsed_json, dict):
                        nima_data = parsed_json.get("nima_scores")
                        if isinstance(nima_data, dict):
                            label = (
                                nima_data.get("distribution")
                                or nima_data.get("scores")
                                or nima_data.get("mean_score")
                            )
                        if label is None:
                            label = (
                                parsed_json.get("distribution")
                                or parsed_json.get("scores")
                                or parsed_json.get("label")
                                or parsed_json.get("score")
                            )
                except (OSError, tarfile.TarError, ValueError) as exc:
                    logger.warning(
                        "Unreadable JSON sidecar %s in %s; metadata skipped: %s",
                        json_m,
                        t_path,
                        exc,
                    )

        return Sample(
            name=stem,
            image_bytes=image_bytes,
            image_format=fmt,
            target_bytes=target_bytes,
            mask_bytes=mask_bytes,
            label=label,
            metadata=meta,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Exclude open file handles when pickling across process boundaries."""
        state = self.__dict__.copy()
        state["_tar_handles"] = {}
        state["_tar_stack"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reconstruct state with clean handle cache in worker processes."""
        self.__dict__.update(state)
        self._tar_handles = {}
        self._tar_stack = ExitStack()

    def close(self) -> None:
        """Close all open tar file handles."""
        try:
            self._tar_stack.close()
        except OSError as exc:
            logger.debug("Error closing tarfile: %s", exc)
        self._tar_handles.clear()
        self._tar_stack = ExitStack()
