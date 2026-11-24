"""Metric trajectory vault, moving averages, and plateau detection."""

import csv
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.checkpoint.manager import (
    CHECKPOINT_LOAD_ERRORS,
    _resilient_replace,
    safe_load_checkpoint,
)
from training.utils.atomic import remove_temp_file_safely

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
        val = metrics.get(
            self.primary_metric, float("inf") if self.mode == "min" else float("-inf")
        )

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

    def has_plateaued(self, patience: int = 5) -> bool:
        """Return True when the primary metric set no new best in the last `patience` epochs."""
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
        """Export metric history to CSV format crash-safely with temp-file rename."""
        if not self.history:
            return

        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        all_keys: set[str] = set()
        for r in self.history:
            all_keys.update(r.metrics.keys())
        fieldnames = ["epoch", "timestamp", "is_best", "checkpoint_path"] + sorted(all_keys)

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix="history_", suffix=".tmp", dir=str(target.parent)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f:
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
                f.flush()
                os.fsync(f.fileno())
            _resilient_replace(tmp_path, target)
        except OSError as exc:
            logger.error("Failed exporting metrics to %s: %s", target, exc)
        finally:
            remove_temp_file_safely(tmp_path)

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
                    ckpt_path = row.get("checkpoint_path") or None
                    metrics: dict[str, float] = {}
                    for k, v in row.items():
                        if (
                            k not in {"epoch", "timestamp", "is_best", "checkpoint_path"}
                            and v != ""
                        ):
                            try:
                                metrics[k] = float(v)
                            except ValueError as conv_err:
                                logger.debug(
                                    "Skipping non-numeric metric %s=%r in %s: %s",
                                    k,
                                    v,
                                    target,
                                    conv_err,
                                )
                    self.record_epoch(epoch, metrics, checkpoint_path=ckpt_path)
        except (OSError, csv.Error, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Error reading metrics CSV from %s: %s", target, exc)


@dataclass(frozen=True)
class VaultConsistencyReport:
    """Report emitted by validate_vault_consistency."""

    is_valid: bool
    checkpoints_found: list[str]
    missing_targets: list[str]
    stale_checkpoints: list[str]
    corrupted_checkpoints: list[str]
    issues: list[str]


def _audit_checkpoint_file(
    item: Path,
    checkpoints_found: list[str],
    corrupted: list[str],
    issues: list[str],
) -> None:
    checkpoints_found.append(item.name)
    if item.stat().st_size == 0:
        corrupted.append(item.name)
        issues.append(f"Checkpoint {item.name} is 0 bytes (truncated/empty).")
        return
    try:
        payload = safe_load_checkpoint(item, map_location="cpu")
        if not isinstance(payload, dict):
            corrupted.append(item.name)
            issues.append(f"Checkpoint {item.name} payload is not a dictionary.")
            return
        has_weights = any(
            k in payload and isinstance(payload[k], dict)
            for k in ("model_state", "model_state_dict", "state_dict", "model")
        )
        if not has_weights:
            corrupted.append(item.name)
            issues.append(f"Checkpoint {item.name} lacks model weight state dictionary.")
    except CHECKPOINT_LOAD_ERRORS as exc:
        corrupted.append(item.name)
        issues.append(f"Failed loading checkpoint {item.name}: {exc}")


def _check_stale_progress(ckpt_path: Path, stale: list[str], issues: list[str]) -> None:
    latest_p = ckpt_path / "latest.pth"
    progress_p = ckpt_path / "progress.pth"
    if latest_p.exists() and progress_p.exists():
        try:
            l_data = safe_load_checkpoint(latest_p, map_location="cpu")
            p_data = safe_load_checkpoint(progress_p, map_location="cpu")
            if int(l_data.get("epoch", 0)) >= int(p_data.get("epoch", 0)):
                stale.append(progress_p.name)
                issues.append(
                    f"Stale progress checkpoint {progress_p.name} persists while "
                    "latest.pth is at equal or newer epoch."
                )
        except CHECKPOINT_LOAD_ERRORS as e:
            logger.debug("Notice inspecting latest vs progress: %s", e)


def validate_vault_consistency(
    model_name: str,
    checkpoints_dir: Path | str,
    sota_targets: dict[str, float] | None = None,
) -> VaultConsistencyReport:
    """Audit and validate the consistency of vault checkpoints for a model.

    Verifies:
    1. Readability and non-emptiness of all checkpoints (best, latest, progress, vault_*).
    2. Payload integrity (loads with safe_load_checkpoint, contains model weights).
    3. Consistency with sota_targets (checks for missing or unachieved targets).
    4. Detection of corrupt, truncated, or stale artifacts.
    """
    del model_name
    ckpt_path = Path(checkpoints_dir).resolve()
    if not ckpt_path.exists():
        return VaultConsistencyReport(
            is_valid=False,
            checkpoints_found=[],
            missing_targets=list((sota_targets or {}).keys()),
            stale_checkpoints=[],
            corrupted_checkpoints=[],
            issues=[f"Checkpoints directory does not exist: {ckpt_path}"],
        )

    checkpoints_found: list[str] = []
    corrupted: list[str] = []
    stale: list[str] = []
    issues: list[str] = []

    for item in ckpt_path.glob("*.pth"):
        if item.is_file():
            _audit_checkpoint_file(item, checkpoints_found, corrupted, issues)

    _check_stale_progress(ckpt_path, stale, issues)

    missing_targets = [
        target
        for target in (sota_targets or {})
        if f"vault_{target.lower().strip().replace('-', '_')}.pth" not in checkpoints_found
    ]

    is_valid = len(corrupted) == 0 and len(stale) == 0
    return VaultConsistencyReport(
        is_valid=is_valid,
        checkpoints_found=sorted(checkpoints_found),
        missing_targets=missing_targets,
        stale_checkpoints=sorted(stale),
        corrupted_checkpoints=sorted(corrupted),
        issues=issues,
    )


def prune_vault_checkpoints(checkpoints_dir: Path | str, dry_run: bool = False) -> list[str]:
    """Prune stale, redundant, and temporary files from the checkpoint vault.

    Removes:
    1. Stray '.tmp' files left from interrupted saves.
    2. Stale progress.pth when latest.pth represents an equal or newer epoch.
    Returns list of deleted/pruned file names.
    """
    ckpt_path = Path(checkpoints_dir).resolve()
    if not ckpt_path.exists():
        return []

    pruned: list[str] = []

    # 1. Prune temporary files
    for item in ckpt_path.glob("*.tmp"):
        if item.is_file():
            pruned.append(item.name)
            if not dry_run:
                try:
                    item.unlink()
                    logger.info("Pruned temporary file: %s", item.name)
                except OSError as e:
                    logger.warning("Could not unlink temp file %s: %s", item.name, e)

    # 2. Prune obsolete progress.pth
    latest_p = ckpt_path / "latest.pth"
    progress_p = ckpt_path / "progress.pth"
    if latest_p.exists() and progress_p.exists():
        try:
            l_data = safe_load_checkpoint(latest_p, map_location="cpu")
            p_data = safe_load_checkpoint(progress_p, map_location="cpu")
            if int(l_data.get("epoch", 0)) >= int(p_data.get("epoch", 0)):
                pruned.append(progress_p.name)
                if not dry_run:
                    progress_p.unlink()
                    logger.info(
                        "Pruned obsolete progress checkpoint %s (latest ep %d >= progress ep %d)",
                        progress_p.name,
                        int(l_data.get("epoch", 0)),
                        int(p_data.get("epoch", 0)),
                    )
        except CHECKPOINT_LOAD_ERRORS as e:
            logger.debug("Notice while checking progress pruning: %s", e)

    return pruned
