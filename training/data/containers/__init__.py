"""Container reader plugin ecosystem and auto-resolution factory."""

import logging
from pathlib import Path
from typing import Any
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


def resolve_container_reader(
    manifold_path: Path | str,
    split: str = "train",
) -> ContainerReader:
    """Auto-detect and instantiate the appropriate ContainerReader for a dataset manifold."""
    root = Path(manifold_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cannot resolve container reader for nonexistent path: {root}")

    # 1. Check for dataset_info.yaml configuration
    info_file = root / "dataset_info.yaml"
    primary_format = ""
    if info_file.exists():
        try:
            info_data = yaml.safe_load(info_file.read_text(encoding="utf-8")) or {}
            if isinstance(info_data, dict):
                container_cfg = info_data.get("container", {})
                if isinstance(container_cfg, dict):
                    primary_format = container_cfg.get("primary", "").lower()
        except Exception as exc:
            logger.debug("Error reading %s: %s", info_file, exc)

    if primary_format == "directory":
        return DirectoryReader(root, split=split)
    if primary_format == "parquet":
        return ParquetReader(root, split=split)
    if primary_format == "mds":
        return MdsReader(root, split=split)
    if primary_format == "litdata":
        return LitDataReader(root, split=split)
    if primary_format == "webdataset":
        return WebDatasetReader(root, split=split)

    # 2. Heuristic inspection based on directory contents
    split_dir = root / split
    check_dir = split_dir if split_dir.exists() and split_dir.is_dir() else root

    # Check for parquet
    if root.is_file() and root.suffix == ".parquet":
        return ParquetReader(root, split=split)
    if list(check_dir.glob("*.parquet")):
        return ParquetReader(root, split=split)

    # Check for webdataset tar files
    if list(check_dir.glob("*.tar")):
        return WebDatasetReader(root, split=split)

    # Check for MDS index.json
    if (check_dir / "index.json").exists() and not (check_dir / "images").exists():
        return MdsReader(root, split=split)

    # Check for LitData chunks
    if list(check_dir.glob("*.bin")) and not (check_dir / "images").exists():
        return LitDataReader(root, split=split)

    # Default to standard DirectoryReader
    return DirectoryReader(root, split=split)
