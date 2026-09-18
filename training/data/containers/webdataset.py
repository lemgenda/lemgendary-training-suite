"""WebDataset container reader for sharded tar archives."""

import io
import logging
from pathlib import Path
import tarfile
from typing import Any

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.webdataset")

IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}


class WebDatasetReader:
    """Reader for tar-sharded WebDataset archives with standard library tarfile fallback."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate tar archives
        split_dir = self.root / self.split
        search_dir = split_dir if split_dir.exists() and split_dir.is_dir() else self.root

        if search_dir.is_file() and search_dir.suffix == ".tar":
            self.tar_files = [search_dir]
        else:
            self.tar_files = sorted(search_dir.glob("*.tar"))

        if not self.tar_files:
            raise FileNotFoundError(f"No WebDataset tar archives found in '{search_dir}'.")

        # Index members across tar archives
        # Map sample index -> (tar_path, image_member_name, target_member_name)
        self.index_map: list[tuple[Path, str, str | None]] = []
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

        for t_path in self.tar_files:
            try:
                tf = tarfile.open(t_path, mode="r:*")
                self._tar_handles[t_path] = tf
                members = tf.getmembers()
                # Group by stem
                stems: dict[str, dict[str, tarfile.TarInfo]] = {}
                for m in members:
                    if not m.isfile():
                        continue
                    m_path = Path(m.name)
                    ext = m_path.suffix.lower()
                    stem = m_path.stem
                    if stem not in stems:
                        stems[stem] = {}
                    stems[stem][ext] = m

                for stem, exts in stems.items():
                    # Check for image member
                    img_info = None
                    for iext in IMAGE_EXTENSIONS:
                        if iext in exts:
                            img_info = exts[iext]
                            break
                    if img_info is not None:
                        tgt_info = None
                        # Target could be .target.png, or in a targets group
                        self.index_map.append((t_path, img_info.name, tgt_info.name if tgt_info else None))
            except Exception as exc:
                logger.warning("Failed opening tar file %s: %s", t_path, exc)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.index_map):
            raise IndexError(f"WebDataset index {index} out of range (0..{len(self.index_map) - 1}).")

        t_path, img_member, tgt_member = self.index_map[index]
        tf = self._tar_handles.get(t_path)
        if tf is None:
            tf = tarfile.open(t_path, mode="r:*")
            self._tar_handles[t_path] = tf

        extracted = tf.extractfile(img_member)
        image_bytes = extracted.read() if extracted is not None else b""
        stem = Path(img_member).stem
        fmt = Path(img_member).suffix.lstrip(".").lower()

        target_bytes: bytes | None = None
        if tgt_member is not None:
            tgt_extracted = tf.extractfile(tgt_member)
            if tgt_extracted is not None:
                target_bytes = tgt_extracted.read()

        return Sample(
            name=stem,
            image_bytes=image_bytes,
            image_format=fmt,
            target_bytes=target_bytes,
            metadata={"tar_file": str(t_path), "member": img_member},
        )

    def close(self) -> None:
        """Close all open tar file handles."""
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception as exc:
                logger.debug("Error closing tarfile: %s", exc)
        self._tar_handles.clear()
