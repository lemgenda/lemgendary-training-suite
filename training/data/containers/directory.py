"""Directory-based container reader supporting lossless WebP, PNG, JPEG, and NTFS hardlinks."""

import io
import logging
from pathlib import Path
from typing import Any
from PIL import Image

from training.data.containers.base import Sample

logger = logging.getLogger("lemtrain.containers.directory")

SUPPORTED_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg"}


class DirectoryReader:
    """Reads directory-based datasets supporting modern lossless WebP and traditional images."""

    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root).resolve()
        self.split = split

        # Locate image directory
        self.img_dir = self._resolve_subfolder("images")
        if self.img_dir is None or not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found in manifold '{self.root}' for split '{self.split}'.")

        # Locate optional directories
        self.tgt_dir = self._resolve_subfolder("targets")
        self.mask_dir = self._resolve_subfolder("masks")
        self.lbl_dir = self._resolve_subfolder("labels")

        # Index samples
        self.sample_files: list[Path] = sorted([
            f for f in self.img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
        ])

    def _resolve_subfolder(self, folder_name: str) -> Path | None:
        """Resolve split-aware folder or root folder."""
        # 1. <root>/<folder>/<split>
        split_cand = self.root / folder_name / self.split
        if split_cand.exists() and split_cand.is_dir():
            return split_cand

        # 2. <root>/<split>/<folder>
        inv_cand = self.root / self.split / folder_name
        if inv_cand.exists() and inv_cand.is_dir():
            return inv_cand

        # 3. <root>/<folder>
        flat_cand = self.root / folder_name
        if flat_cand.exists() and flat_cand.is_dir():
            return flat_cand

        return None

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.sample_files):
            raise IndexError(f"Sample index {index} out of range (0..{len(self.sample_files) - 1}).")

        img_path = self.sample_files[index]
        stem = img_path.stem
        fmt = img_path.suffix.lstrip(".").lower()

        image_bytes = img_path.read_bytes()

        # Target bytes
        target_bytes: bytes | None = None
        if self.tgt_dir is not None:
            for ext in [img_path.suffix, ".png", ".webp", ".jpg", ".jpeg"]:
                tgt_candidate = self.tgt_dir / f"{stem}{ext}"
                if tgt_candidate.exists():
                    target_bytes = tgt_candidate.read_bytes()
                    break

        # Mask bytes
        mask_bytes: bytes | None = None
        if self.mask_dir is not None:
            for ext in [".png", ".webp", ".jpg", ".jpeg"]:
                mask_candidate = self.mask_dir / f"{stem}{ext}"
                if mask_candidate.exists():
                    mask_bytes = mask_candidate.read_bytes()
                    break

        # Label data
        label: Any = None
        if self.lbl_dir is not None:
            lbl_candidate = self.lbl_dir / f"{stem}.txt"
            if lbl_candidate.exists():
                try:
                    label = lbl_candidate.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    logger.debug("Failed reading label for %s: %s", stem, exc)

        return Sample(
            name=stem,
            image_bytes=image_bytes,
            image_format=fmt,
            target_bytes=target_bytes,
            mask_bytes=mask_bytes,
            label=label,
            metadata={"path": str(img_path)},
        )

    @staticmethod
    def decode_image(image_bytes: bytes) -> Image.Image:
        """Decode image bytes to PIL Image RGB."""
        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        return img.convert("RGB")

    def close(self) -> None:
        """DirectoryReader does not hold persistent file handles."""
        pass
