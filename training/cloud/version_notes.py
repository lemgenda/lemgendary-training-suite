"""Version notes of a Kaggle Models Hub upload: one builder for every publisher (K-MDL-5).

The GUI and CLI push, the per-epoch auto-sync and the notebook release all call
:func:`build_version_notes`, so every Hub version says which epoch it holds, the last
``metrics.csv`` values and the suite version that produced it, for example::

    Release: mirnet_exposure [ONNX FP16] | epoch 41 | val_loss=0.0123 psnr=27.1000 | suite v16.9.177
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from training.cloud.variations import CHECKPOINTS_DIR
from training.fid_reeval.ledger import filter_metrics_row_for_fid_provenance
from training.utils.model_payload import checkpoint_epoch
from training.version import __version__

logger = logging.getLogger("lemtrain.cloud.version_notes")

METRICS_FILE = "metrics.csv"
NOTES_METRIC_LIMIT = 6
"""At most this many metric values are written into one version note."""
CHECKPOINT_PRIORITY: tuple[str, ...] = ("latest.pth", "progress.pth", "best.pth")
"""Checkpoint whose stored epoch a note reports (first one present)."""
_EPOCH_COLUMNS = ("epoch", "Epoch")
_NON_METRIC_COLUMNS = frozenset({*_EPOCH_COLUMNS, "phase", "fold", "lr", "train_loss"})


@dataclass(frozen=True)
class ReleaseFacts:
    """What a model folder holds: the checkpoint epoch and the last metrics row."""

    epoch: int | None
    metrics: dict[str, str] = field(default_factory=dict)


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def last_metrics(model_dir: Path) -> tuple[str | None, dict[str, str]]:
    """Epoch and measured values of the last ``metrics.csv`` row (FID provenance applied)."""
    path = model_dir / METRICS_FILE
    if not path.is_file():
        return None, {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        logger.warning("Cannot read %s for version notes: %s", path, exc)
        return None, {}
    if not rows:
        return None, {}
    row = filter_metrics_row_for_fid_provenance(rows[-1], model_dir)
    epoch = next((str(row[k]).strip() for k in _EPOCH_COLUMNS if row.get(k)), None)
    metrics: dict[str, str] = {}
    for name, value in row.items():
        text = str(value or "").strip()
        if name and name not in _NON_METRIC_COLUMNS and text and _finite(text):
            metrics[name] = text
        if len(metrics) >= NOTES_METRIC_LIMIT:
            break
    return epoch, metrics


def release_facts(model_dir: Path, epoch: int | None = None) -> ReleaseFacts:
    """Epoch and metrics of ``model_dir``.

    ``epoch`` when the caller knows it (the per-epoch auto-sync), else the epoch stored in the
    newest checkpoint, else the last metrics row's.
    """
    csv_epoch, metrics = last_metrics(model_dir)
    if epoch is not None:
        return ReleaseFacts(epoch, metrics)
    for name in CHECKPOINT_PRIORITY:
        path = model_dir / CHECKPOINTS_DIR / name
        if path.is_file():
            epoch = checkpoint_epoch(path)
            if epoch is not None:
                return ReleaseFacts(epoch, metrics)
    epoch_from_csv = int(float(csv_epoch)) if csv_epoch and _finite(csv_epoch) else None
    return ReleaseFacts(epoch_from_csv, metrics)


def build_version_notes(
    model_key: str, facts: ReleaseFacts, *, label: str, reason: str = "Release"
) -> str:
    """Notes of one upload of ``model_key`` (``label``: the variation label)."""
    parts = [f"{reason}: {model_key} [{label}]"]
    parts.append(f"epoch {facts.epoch}" if facts.epoch is not None else "epoch unknown")
    if facts.metrics:
        parts.append(" ".join(f"{k}={v}" for k, v in facts.metrics.items()))
    parts.append(f"suite v{__version__}")
    return " | ".join(parts)
