"""Metric trajectory vault, moving averages, and plateau detection."""

import csv
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger("lemtrain.checkpoint.vault")


@dataclass
class MetricRecord:
    """Individual epoch metric snapshot."""

    epoch: int
    metrics: dict[str, float]
    primary_metric: str
    primary_value: float
    is_best: bool
    timestamp: float = field(default_factory=time.time)
    checkpoint_path: str | None = None


class MetricVault:
    """Tracks training and validation metric history, evaluates best SOTA, and detects plateaus."""

    def __init__(self, primary_metric: str = "val_loss", mode: str = "min") -> None:
        self.primary_metric = primary_metric
        self.mode = mode.lower()
        if self.mode not in {"min", "max"}:
            raise ValueError(f"MetricVault mode must be 'min' or 'max', got '{mode}'.")

        self.history: list[MetricRecord] = []
        self.best_record: MetricRecord | None = None

    def _is_better(self, new_val: float, baseline_val: float) -> bool:
        """Evaluate if new_val improves upon baseline_val according to mode."""
        if self.mode == "min":
            return new_val < baseline_val
        return new_val > baseline_val

    def record_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
        checkpoint_path: Path | str | None = None,
    ) -> bool:
        """Record an epoch snapshot, evaluating whether it sets a new best primary metric."""
        val = metrics.get(self.primary_metric, float("inf") if self.mode == "min" else float("-inf"))

        is_best = False
        if self.best_record is None or self._is_better(val, self.best_record.primary_value):
            is_best = True

        record = MetricRecord(
            epoch=epoch,
            metrics=metrics,
            primary_metric=self.primary_metric,
            primary_value=val,
            is_best=is_best,
            checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        )
        self.history.append(record)

        if is_best:
            self.best_record = record
            logger.info("New best %s achieved at epoch %d: %.4f", self.primary_metric, epoch, val)

        return is_best

    def get_rolling_average(self, metric_name: str, window: int = 5) -> float | None:
        """Compute rolling mean of a metric across the last `window` recorded epochs."""
        if not self.history:
            return None

        slice_records = self.history[-window:]
        vals = [r.metrics[metric_name] for r in slice_records if metric_name in r.metrics]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def has_plateaued(self, patience: int = 5, min_delta: float = 1e-4) -> bool:
        """Detect whether the primary metric has failed to improve for `patience` consecutive epochs."""
        if len(self.history) < patience + 1 or self.best_record is None:
            return False

        last_patience = self.history[-patience:]
        best_in_window = False
        for rec in last_patience:
            if rec.epoch == self.best_record.epoch:
                best_in_window = True
                break

        return not best_in_window

    def export_csv(self, path: Path | str) -> None:
        """Export metric history to CSV format."""
        if not self.history:
            return

        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        all_keys: set[str] = set()
        for r in self.history:
            all_keys.update(r.metrics.keys())
        fieldnames = ["epoch", "timestamp", "is_best", "checkpoint_path"] + sorted(all_keys)

        try:
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.history:
                    row: dict[str, Any] = {
                        "epoch": r.epoch,
                        "timestamp": r.timestamp,
                        "is_best": r.is_best,
                        "checkpoint_path": r.checkpoint_path or "",
                    }
                    row.update(r.metrics)
                    writer.writerow(row)
        except OSError as exc:
            logger.error("Failed exporting metrics to %s: %s", target, exc)

    def load_csv(self, path: Path | str) -> None:
        """Load history from a CSV log file."""
        target = Path(path).resolve()
        if not target.exists():
            return

        try:
            with open(target, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(row.get("epoch", 0))
                    is_best = row.get("is_best", "False").lower() in {"true", "1"}
                    ckpt_path = row.get("checkpoint_path") or None
                    metrics: dict[str, float] = {}
                    for k, v in row.items():
                        if k not in {"epoch", "timestamp", "is_best", "checkpoint_path"} and v != "":
                            try:
                                metrics[k] = float(v)
                            except ValueError:
                                pass
                    self.record_epoch(epoch, metrics, checkpoint_path=ckpt_path)
        except Exception as exc:
            logger.warning("Error reading metrics CSV from %s: %s", target, exc)
