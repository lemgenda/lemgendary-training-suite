"""Container reader plugin ecosystem and auto-resolution factory."""

import logging
from pathlib import Path

import yaml

from training.data.containers.base import ContainerReader, Sample
from training.data.containers.directory import DirectoryReader
from training.data.containers.litdata import LitDataReader
from training.data.containers.mds import MdsReader
from training.data.containers.parquet import ParquetReader
from training.data.containers.webdataset import WebDatasetReader

logger = logging.getLogger("lemtrain.containers")

__all__ = [
    "ContainerReader",
    "DirectoryReader",
    "LitDataReader",
    "MdsReader",
    "ParquetReader",
    "Sample",
    "WebDatasetReader",
    "resolve_container_reader",
]


_FORMAT_READERS: dict[str, type[ContainerReader]] = {
    "directory": DirectoryReader,
    "parquet": ParquetReader,
    "mds": MdsReader,
    "litdata": LitDataReader,
    "webdataset": WebDatasetReader,
    "wds": WebDatasetReader,
}


def _declared_format(root: Path) -> str:
    """Return the lower-cased container format named by ``dataset_info.yaml`` ("" if none)."""
    info_file = root / "dataset_info.yaml"
    primary_format = ""
    if info_file.exists():
        try:
            info_data = yaml.safe_load(info_file.read_text(encoding="utf-8")) or {}
            if isinstance(info_data, dict):
                raw_fmt = (
                    info_data.get("format")
                    or info_data.get("canonical_format")
                    or (
                        info_data.get("container", {})
                        if isinstance(info_data.get("container"), dict)
                        else {}
                    ).get("primary", "")
                )
                if raw_fmt:
                    primary_format = str(raw_fmt).strip().lower()
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.warning(
                "Could not read container format from %s; falling back to auto-detection: %s",
                info_file,
                exc,
            )
    return primary_format


def _any_glob(dirs: tuple[Path, ...], patterns: tuple[str, ...]) -> bool:
    """Return True when any of ``dirs`` holds a file matching any of ``patterns``."""
    return any(next(d.glob(pattern), None) is not None for d in dirs for pattern in patterns)


def _detect_reader_class(root: Path, split: str) -> type[ContainerReader]:
    """Pick a reader from the files present (Parquet, WebDataset, MDS, LitData, directory)."""
    split_dir = root / split
    check_dir = split_dir if split_dir.exists() and split_dir.is_dir() else root

    dirs = (check_dir, root)
    if (
        (root.is_file() and root.suffix == ".parquet")
        or (root / "parquet").exists()
        or _any_glob(dirs, ("*.parquet",))
    ):
        return ParquetReader

    # Webdataset shard files (.data or .tar)
    if (
        (root.is_file() and root.suffix in (".data", ".tar"))
        or (root / "shards").exists()
        or _any_glob(dirs, ("*.data", "*.tar"))
    ):
        return WebDatasetReader

    # MDS index.json
    if (root / "mds").exists() or (
        (check_dir / "index.json").exists() and not (check_dir / "images").exists()
    ):
        return MdsReader

    # LitData chunks
    if (root / "litdata").exists() or (
        list(check_dir.glob("*.bin")) and not (check_dir / "images").exists()
    ):
        return LitDataReader

    return DirectoryReader


def resolve_container_reader(
    manifold_path: Path | str,
    split: str = "train",
) -> ContainerReader:
    """Auto-detect and instantiate the appropriate ContainerReader for a dataset manifold.

    A format named in ``dataset_info.yaml`` wins; otherwise the files present decide.
    """
    root = Path(manifold_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cannot resolve container reader for nonexistent path: {root}")

    reader_cls = _FORMAT_READERS.get(_declared_format(root)) or _detect_reader_class(root, split)
    return reader_cls(root, split=split)
