"""Resume-point discovery for the YOLOv8n curriculum governor.

Finds the checkpoint to continue from in the LemGendaryModels hub, reads its optimizer and
epoch state, and replays ``metrics.csv`` to learn which ladder rungs already ran. The
``ResumePoint`` then answers, per rung, whether it is complete, resumes mid-run, or starts
fresh. Nothing here moves resolution or data fraction backwards: completed rungs are only
skipped, never revisited.
"""

from __future__ import annotations

import csv
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from training.checkpoint.manager import CHECKPOINT_LOAD_ERRORS
from training.governance.yolo_ladder import YOLOLadderStage

logger = logging.getLogger("lemtrain.governance.yolo")

# Checkpoint names whose dict payload may carry resumable optimizer/epoch state.
_RESUMABLE_NAMES = ("last.pt", "best.pt", "progress.pth", "best.pth", "yolov8n.pt")


def wipe_clean_run(models_hub_dir: Path, models_hub_ckpt_dir: Path) -> None:
    """Delete prior checkpoints, curriculum state and metrics for a clean run."""
    print(
        "[GOVERNOR] [CLEAN] Clean run requested. Wiping prior checkpoints, state, and metrics...",
        flush=True,
    )
    for p in [
        models_hub_dir / "metrics.csv",
        models_hub_ckpt_dir / "curriculum_state.json",
        models_hub_ckpt_dir / "last.pt",
        models_hub_ckpt_dir / "progress.pth",
        models_hub_ckpt_dir / "best.pt",
        models_hub_ckpt_dir / "best.pth",
    ]:
        if p.exists():
            try:
                p.unlink()
            except OSError as exc:
                logger.warning("Clean run could not remove %s: %s", p, exc)


def discover_checkpoint(models_hub_dir: Path, models_hub_ckpt_dir: Path) -> Path | None:
    """Return the first non-empty hub checkpoint in resume priority order, if any."""
    for c in [
        models_hub_ckpt_dir / "last.pt",
        models_hub_ckpt_dir / "progress.pth",
        models_hub_ckpt_dir / "best.pt",
        models_hub_ckpt_dir / "best.pth",
        models_hub_dir / "yolov8n.pt",
    ]:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def load_resumable_state(candidate_ckpt: Path | None) -> dict[str, Any] | None:
    """Return the checkpoint dict when it carries ``epoch`` and optimizer state."""
    if not (candidate_ckpt and candidate_ckpt.name in _RESUMABLE_NAMES):
        return None
    try:
        try:
            loaded = torch.load(str(candidate_ckpt), map_location="cpu")
        except (pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as safe_err:
            # weights_only=True (torch>=2.6 default) rejects Ultralytics checkpoint classes.
            logger.debug(
                "Restricted load of %s failed, retrying full unpickle: %s",
                candidate_ckpt,
                safe_err,
            )
            loaded = torch.load(str(candidate_ckpt), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "epoch" in loaded and loaded.get("optimizer") is not None:
            return loaded
    except CHECKPOINT_LOAD_ERRORS as exc:
        logger.warning("Failed reading checkpoint dict from %s: %s", candidate_ckpt, exc)
    return None


@dataclass
class MetricsHistory:
    """What a prior run's ``metrics.csv`` says about completed epochs and rungs."""

    csv_epochs: int = 0
    last_res: int | None = None
    last_frac: float | None = None
    stage_epoch_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    def record_row(self, row: dict[str, str]) -> None:
        """Fold one CSV row's epoch, resolution and data fraction into the history."""
        ep = row.get("Epoch")
        if not (ep and ep.isdigit()):
            return
        self.csv_epochs = max(self.csv_epochs, int(ep))
        res_val = row.get("Res")
        data_val = row.get("Data")
        if res_val and res_val.isdigit():
            r_int = int(res_val)
            self.last_res = r_int
            f_pct = round(float(data_val) * 100) if data_val else 100
            self.last_frac = float(data_val) if data_val else 1.0
            key = (r_int, f_pct)
            self.stage_epoch_counts[key] = self.stage_epoch_counts.get(key, 0) + 1


def replay_metrics_history(
    metrics_csv_path: Path,
    best_metrics: dict[str, float],
    best_export_quality: float,
    track_export_quality: bool,
) -> tuple[MetricsHistory, float]:
    """Replay ``metrics.csv`` into a ``MetricsHistory``.

    ``best_metrics`` (``map50`` / ``map50_95``) is raised in place to the best values seen.
    When ``track_export_quality`` is set (an exported model already exists) the returned
    export quality is raised to the best combined score too. A malformed file keeps
    whatever was read before the error.
    """
    history = MetricsHistory()
    try:
        with open(metrics_csv_path, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                history.record_row(row)
                m50 = float(row.get("mAP50", 0.0) or 0.0)
                m95 = float(row.get("mAP50-95", 0.0) or 0.0)
                best_metrics["map50"] = max(best_metrics["map50"], m50)
                best_metrics["map50_95"] = max(best_metrics["map50_95"], m95)
                if track_export_quality:
                    best_export_quality = max(best_export_quality, (m95 * 0.7) + (m50 * 0.3))
    except (OSError, ValueError, csv.Error) as e:
        logger.warning("Error reading %s; resume history incomplete: %s", metrics_csv_path, e)
    return history, best_export_quality


class ResumePoint:
    """Checkpoint state plus metrics history that decide where each ladder rung starts."""

    def __init__(self, ckpt_dict: dict[str, Any] | None, history: MetricsHistory) -> None:
        self.ckpt_dict = ckpt_dict
        self.history = history
        train_args = ckpt_dict.get("train_args", {}) if ckpt_dict else {}
        self.ckpt_epoch = int(ckpt_dict.get("epoch", -1)) if ckpt_dict else -1
        self.ckpt_imgsz = int(train_args.get("imgsz", 0)) if ckpt_dict else 0
        self.ckpt_fraction = float(train_args.get("fraction", 1.0)) if ckpt_dict else 1.0

    def stage_completed(self, stage: YOLOLadderStage) -> bool:
        """Whether the checkpoint (or, without one, the CSV history) is past ``stage``."""
        if self.ckpt_dict:
            return self.ckpt_imgsz > stage.resolution or (
                self.ckpt_imgsz == stage.resolution and self.ckpt_fraction > (stage.fraction + 0.05)
            )
        last_res, last_frac = self.history.last_res, self.history.last_frac
        return last_res is not None and (
            last_res > stage.resolution
            or (
                last_res == stage.resolution
                and last_frac is not None
                and last_frac > (stage.fraction + 0.05)
            )
        )

    def resumes(self, stage: YOLOLadderStage) -> bool:
        """Whether ``stage`` continues mid-run from the loaded checkpoint."""
        return bool(
            self.ckpt_dict
            and self.ckpt_imgsz == stage.resolution
            and abs(self.ckpt_fraction - stage.fraction) <= 0.05
            and self.ckpt_epoch >= 0
        )

    def prior_epochs(self, stage: YOLOLadderStage, frac_pct: int, default: int) -> int:
        """Epochs the CSV history recorded for ``stage`` at ``frac_pct`` percent data."""
        return self.history.stage_epoch_counts.get((stage.resolution, frac_pct), default)
