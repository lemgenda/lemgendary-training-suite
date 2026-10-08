"""WebDataset container reader for sharded tar archives."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import tarfile
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
    parent_parts = [part.lower() for part in p.parent.parts]
    if "targets" in parent_parts:
        return p.stem, "target", ext
    if "masks" in parent_parts:
        return p.stem, "mask", ext
    if "images" in parent_parts:
        return p.stem, "image", ext

    # WebDataset dot-separated pattern:
    # 000001.target.webp -> key='000001', role='target', ext='.webp'
    # 000001.mask.webp   -> key='000001', role='mask', ext='.webp'
    # 000001.webp        -> key='000001', role='image', ext='.webp'
    if len(parts) >= 3:
        key = parts[0]
        sub = parts[1].lower()
        if sub in ("target", "tgt", "gt", "hr"):
            return key, "target", ext
        if sub in ("mask", "msk", "seg"):
            return key, "mask", ext
        return key, sub, ext
    if len(parts) == 2:
        key = parts[0]
        if ext in IMAGE_EXTENSIONS:
            return key, "image", ext
        if ext in (".txt", ".caption"):
            return key, "text", ext
        if ext == ".json":
            return key, "json", ext
        return key, "other", ext
    return p.stem, "image", ext


class WebDatasetReader:
    """Reader for tar-sharded WebDataset archives with support for unpacked shard directories."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate tar archives across candidate locations
        cand_dirs = [
            self.root / "shards" / self.split,
            self.root / self.split / "shards",
            self.root / "shards",
            self.root / self.split,
            self.root,
        ]
        self.tar_files: list[Path] = []
        if self.root.is_file() and self.root.suffix == ".tar":
            self.tar_files = [self.root]
        else:
            for d in cand_dirs:
                if d.exists() and d.is_dir():
                    tars = sorted(d.glob("*.tar"))
                    if tars:
                        self.tar_files = tars
                        break
            if not self.tar_files:
                self.tar_files = sorted(self.root.glob("**/*.tar"))

        # Map sample index -> (storage_mode, tar_path, img_ref, tgt_ref, mask_ref, txt_ref, json_ref)
        self.index_map: list[tuple[str, Path | None, Any, Any, Any, Any, Any]] = []
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

        if self.tar_files:
            for t_path in self.tar_files:
                try:
                    tf = tarfile.open(t_path, mode="r:*")
                    self._tar_handles[t_path] = tf
                    members = tf.getmembers()

                    # Group by sample key -> role -> member_name
                    samples: dict[str, dict[str, str]] = {}
                    for m in members:
                        if not m.isfile():
                            continue
                        key, role, ext = _parse_member_name(m.name)
                        if key not in samples:
                            samples[key] = {}
                        samples[key][role] = m.name

                    for key, roles in samples.items():
                        img_member = roles.get("image")
                        if img_member is not None:
                            tgt_member = roles.get("target")
                            mask_member = roles.get("mask")
                            txt_member = roles.get("text")
                            json_member = roles.get("json")
                            self.index_map.append((
                                "tar",
                                t_path,
                                img_member,
                                tgt_member,
                                mask_member,
                                txt_member,
                                json_member,
                            ))
                except Exception as exc:
                    logger.warning("Failed opening tar file %s: %s", t_path, exc)
        else:
            # Mode B: Extracted shard directories or loose image files
            for d in cand_dirs:
                if not d.exists() or not d.is_dir():
                    continue

                sub_dirs = sorted([sub for sub in d.iterdir() if sub.is_dir()])
                target_dirs = sub_dirs if sub_dirs else [d]

                for scan_dir in target_dirs:
                    samples_dir: dict[str, dict[str, Path]] = {}
                    try:
                        for entry in sorted(scan_dir.iterdir()):
                            if not entry.is_file():
                                continue
                            ext = entry.suffix.lower()
                            if ext in IMAGE_EXTENSIONS or ext in (".txt", ".caption", ".json"):
                                key, role, _ = _parse_member_name(entry.name)
                                if key not in samples_dir:
                                    samples_dir[key] = {}
                                samples_dir[key][role] = entry
                    except OSError as scan_err:
                        logger.debug("Failed scanning directory %s: %s", scan_dir, scan_err)
                        continue

                    for key, roles in samples_dir.items():
                        img_path = roles.get("image")
                        if img_path is not None:
                            tgt_path = roles.get("target")
                            mask_path = roles.get("mask")
                            txt_path = roles.get("text")
                            json_path = roles.get("json")
                            self.index_map.append((
                                "file",
                                None,
                                img_path,
                                tgt_path,
                                mask_path,
                                txt_path,
                                json_path,
                            ))

                if self.index_map:
                    break

        if not self.index_map and self.split == "train":
            raise FileNotFoundError(
                f"No WebDataset tar archives or extracted shard directories found in '{self.root}' for split '{self.split}'."
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.index_map):
            raise IndexError(f"WebDataset index {index} out of range (0..{len(self.index_map) - 1}).")

        entry = self.index_map[index]
        mode = entry[0]

        if mode == "file":
            _, _, img_path, tgt_path, mask_path, txt_path, json_path = entry
            image_bytes = img_path.read_bytes()
            stem = img_path.stem.split(".")[0]
            fmt = img_path.suffix.lstrip(".").lower()

            target_bytes = tgt_path.read_bytes() if tgt_path is not None and tgt_path.exists() else None
            mask_bytes = mask_path.read_bytes() if mask_path is not None and mask_path.exists() else None

            label: Any = None
            meta: dict[str, Any] = {"file": str(img_path)}

            if txt_path is not None and txt_path.exists():
                try:
                    label = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            if json_path is not None and json_path.exists():
                try:
                    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
                    meta["json"] = parsed_json
                    if label is None and isinstance(parsed_json, dict):
                        label = (
                            parsed_json.get("distribution")
                            or parsed_json.get("scores")
                            or parsed_json.get("label")
                            or parsed_json.get("score")
                        )
                except Exception:
                    pass

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
        tf = self._tar_handles.get(t_path)
        if tf is None:
            tf = tarfile.open(t_path, mode="r:*")
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
                except Exception:
                    pass

        if json_m is not None:
            json_extracted = tf.extractfile(json_m)
            if json_extracted is not None:
                try:
                    parsed_json = json.loads(json_extracted.read().decode("utf-8"))
                    meta["json"] = parsed_json
                    if label is None and isinstance(parsed_json, dict):
                        label = (
                            parsed_json.get("distribution")
                            or parsed_json.get("scores")
                            or parsed_json.get("label")
                            or parsed_json.get("score")
                        )
                except Exception:
                    pass

        return Sample(
            name=stem,
            image_bytes=image_bytes,
            image_format=fmt,
            target_bytes=target_bytes,
            mask_bytes=mask_bytes,
            label=label,
            metadata=meta,
        )

    def close(self) -> None:
        """Close all open tar file handles."""
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception as exc:
                logger.debug("Error closing tarfile: %s", exc)
        self._tar_handles.clear()
