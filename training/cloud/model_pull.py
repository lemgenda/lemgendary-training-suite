"""Merge a downloaded Kaggle model variation into a local hub folder without losing local work.

A pull never downloads into the hub folder itself: kagglehub refuses a non-empty
``output_dir`` and, with ``force_download``, empties it first. The caller downloads into a
fresh temporary folder and this module merges the files into the hub folder one by one:

* a file missing locally is added;
* with ``force`` the pulled files replace their local copies (only those files; nothing else
  in the folder is touched or deleted);
* otherwise a checkpoint replaces its local copy only when its stored epoch is newer
  (:func:`training.utils.model_payload.checkpoint_epoch`, never the file size), and any other
  pulled file replaces its local copy only when the pulled checkpoints are newer than the
  local ones. A payload without checkpoints (``.pt`` / ONNX exports) keeps existing local
  files unless forced.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from training.cloud.variations import CHECKPOINT_SUFFIX
from training.utils.model_payload import checkpoint_epoch

logger = logging.getLogger("lemtrain.cloud.model_pull")

EpochReader = Callable[[Path], "int | None"]
PathFilter = Callable[[Path], bool]


@dataclass
class MergeReport:
    """Relative paths written into the hub folder and local files kept."""

    written: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


def _accept_all(_relative: Path) -> bool:
    return True


def _relative_files(root: Path, keep: PathFilter) -> list[Path]:
    files = (p.relative_to(root) for p in root.rglob("*") if p.is_file())
    return sorted(rel for rel in files if keep(rel))


def _place(src: Path, dst: Path) -> None:
    """Move ``src`` onto ``dst`` (copy when they are on different volumes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as exc:
        logger.debug("Cannot move %s onto %s (%s); copying instead", src, dst, exc)
        shutil.copy2(src, dst)


def _memoized(reader: EpochReader) -> EpochReader:
    """``reader`` that reads each checkpoint's epoch at most once."""
    epochs: dict[Path, int | None] = {}

    def read(path: Path) -> int | None:
        if path not in epochs:
            epochs[path] = reader(path)
        return epochs[path]

    return read


def _newest(epochs: list[int | None]) -> int | None:
    known = [e for e in epochs if e is not None]
    return max(known) if known else None


def _is_newer(remote: int | None, local: int | None) -> bool:
    return remote is not None and (local is None or remote > local)


def merge_downloaded_payload(
    download_dir: Path,
    dest_dir: Path,
    *,
    force: bool,
    keep: PathFilter | None = None,
    epoch_of: EpochReader = checkpoint_epoch,
) -> MergeReport:
    """Merge the files of ``download_dir`` that ``keep`` accepts (default all) into ``dest_dir``."""
    epochs = _memoized(epoch_of)
    files = _relative_files(download_dir, keep or _accept_all)
    checkpoints = [rel for rel in files if rel.suffix == CHECKPOINT_SUFFIX]
    payload_newer = _is_newer(
        _newest([epochs(download_dir / rel) for rel in checkpoints]),
        _newest([epochs(dest_dir / rel) for rel in checkpoints if (dest_dir / rel).is_file()]),
    )

    report = MergeReport()
    for rel in files:
        src, dst = download_dir / rel, dest_dir / rel
        if dst.exists() and not force:
            if rel.suffix == CHECKPOINT_SUFFIX:
                wins = _is_newer(epochs(src), epochs(dst))
            else:
                wins = payload_newer
            if not wins:
                report.kept.append(rel.as_posix())
                continue
        _place(src, dst)
        report.written.append(rel.as_posix())
    logger.info(
        "Merged pull into %s: %d written, %d local kept",
        dest_dir,
        len(report.written),
        len(report.kept),
    )
    return report
