"""Universal Checkpoint Lifecycle Manager for LemGendary AI Training Suite.

Enforces strict 4-tier checkpoint hierarchy, intra-epoch timer bounds,
vault milestone checkpoints, and zero file pollution in lemgendary-training-suite.

Checkpoint Hierarchy:
1. progress.pth: Intra-epoch checkpoint saved every 15 minutes during training
   AND validation phases. Strictly bounded to [5%, 50%] progress within the phase.
   Never saved below 5% or above 50% regardless of time elapsed.
2. latest.pth: Saved at the completion of each epoch after both training and validation
   passes finish. Immediately purges progress.pth upon creation.
3. best.pth: Saved whenever model achieves a new best Quality_Score.
   Immediately triggers universal tri-format export (.onnx FP16, _FP32.onnx + .onnx.data, .pt)
   and refreshes LemGendaryModels/[ModelName]/README.md.
4. vault_[target].pth: Saved whenever a specific target metric defined in
   sota_targets (e.g., psnr, ssim, plcc, map50) achieves a new historical best.

All assets persist exclusively to LemGendaryModels/[ModelName]/ and NEVER to
lemgendary-training-suite/.
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import nn

from training.checkpoint.manager import (
    CHECKPOINT_LOAD_ERRORS,
    safe_atomic_save,
    safe_load_checkpoint,
)
from training.checkpoint.preflight_generator import (
    PREFLIGHT_ERRORS,
    generate_preflight_assets,
)
from training.export.base import EXPORT_ERRORS
from training.export.universal_exporter import (
    export_universal_model,
    get_model_hub_dir,
)
from training.model_card import build_model_readme
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.checkpoint.lifecycle")


LOWER_IS_BETTER_METRICS: set[str] = {
    "loss",
    "val_loss",
    "eval_loss",
    "perplexity",
    "train_loss",
    "lpips",
    "fid",
    "mae",
    "mse",
    "tp_mae",
    "sl_mae",
    "max_drawdown",
    "maxdd",
    "drawdown",
    "rank_margin",
    "entropy",
    "dir_entropy",
}

CANONICAL_METRIC_ALIASES: dict[str, str] = {
    "evalloss": "eval_loss",
    "eval_loss": "eval_loss",
    "valloss": "eval_loss",
    "val_loss": "eval_loss",
    "perplexity": "perplexity",
    "tokenaccuracy": "token_accuracy",
    "token_accuracy": "token_accuracy",
    "diracc": "dir_acc",
    "dir_acc": "dir_acc",
    "winrate": "win_rate",
    "win_rate": "win_rate",
    "profitfactor": "profit_factor",
    "profit_factor": "profit_factor",
    "sharpe": "sharpe_ratio",
    "sharperatio": "sharpe_ratio",
    "sharpe_ratio": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "sortinoratio": "sortino_ratio",
    "sortino_ratio": "sortino_ratio",
    "maxdd": "max_drawdown",
    "max_drawdown": "max_drawdown",
    "tpmae": "tp_mae",
    "tp_mae": "tp_mae",
    "slmae": "sl_mae",
    "sl_mae": "sl_mae",
    "direntropy": "dir_entropy",
    "dir_entropy": "dir_entropy",
    "map_50": "map50",
    "map50": "map50",
    "map_50_95": "map50_95",
    "map50_95": "map50_95",
    "map50-95": "map50_95",
}


def normalize_metric_name(name: str) -> str:
    """Normalize metric key to canonical lower-cased snake_case representation."""
    cleaned = name.strip().lower().replace("-", "_")
    unscored = cleaned.replace("_", "")
    return CANONICAL_METRIC_ALIASES.get(cleaned, CANONICAL_METRIC_ALIASES.get(unscored, cleaned))


def is_lower_better(metric_name: str) -> bool:
    """Determine whether a metric improves when its value decreases."""
    norm = normalize_metric_name(metric_name)
    if norm in LOWER_IS_BETTER_METRICS:
        return True
    return any(sub in norm for sub in ("loss", "mae", "mse", "drawdown", "lpips", "fid"))


STAGING_DIR_NAMES = (".kaggle_upload_stage", ".export_staging")
"""Transient staging folders that :func:`canonicalize_checkpoints` never touches."""


def _in_staging_dir(pth: Path) -> bool:
    return any(parent.name in STAGING_DIR_NAMES for parent in pth.resolve().parents)


def _canonical_checkpoint_name(name_lower: str) -> str | None:
    """Canonical file name for a lower-cased checkpoint name, or None if it is not one."""
    if "vault_" in name_lower:
        metric_part = name_lower.split("vault_")[-1].replace(".pth", "")
        metric_clean = normalize_metric_name(metric_part)
        return f"vault_{metric_clean}.pth"
    if "best" in name_lower:
        return "best.pth"
    if "latest" in name_lower or "last" in name_lower:
        return "latest.pth"
    if "progress" in name_lower:
        return "progress.pth"
    return None


def _move_to_canonical(pth: Path, target_path: Path) -> str | None:
    """Move ``pth`` to ``target_path``; returns the action message, None on failure."""
    try:
        shutil.move(str(pth), str(target_path))
    except OSError as move_err:
        logger.warning("Failed renaming checkpoint %s -> %s: %s", pth, target_path, move_err)
        return None
    msg = f"Renamed legacy checkpoint {pth.name} -> {target_path.name}"
    logger.info(msg)
    return msg


def _purge_duplicate(pth: Path, target_path: Path) -> str | None:
    """Delete ``pth`` since ``target_path`` exists; returns the action message, None on failure."""
    try:
        pth.unlink()
    except OSError as unlink_err:
        logger.debug("Failed unlinking duplicate checkpoint %s: %s", pth, unlink_err)
        return None
    msg = f"Purged redundant duplicate checkpoint {pth.name} (canonical {target_path.name} exists)"
    logger.info(msg)
    return msg


def canonicalize_checkpoints(model_hub_dir: Path, checkpoints_dir: Path) -> list[str]:
    """Rename legacy-named checkpoints in a model directory to canonical names when missing.

    Canonical format:
    - best.pth
    - latest.pth
    - progress.pth
    - vault_[metric].pth
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []

    if not model_hub_dir.exists():
        return actions

    for pth in list(model_hub_dir.rglob("*.pth")):
        # Strictly ignore transient staging directories
        if not pth.is_file() or _in_staging_dir(pth):
            continue

        target_name = _canonical_checkpoint_name(pth.name.lower())
        if not target_name:
            continue

        target_path = checkpoints_dir / target_name
        if pth.resolve() == target_path.resolve():
            continue

        if not target_path.exists():
            msg = _move_to_canonical(pth, target_path)
        else:
            msg = _purge_duplicate(pth, target_path)
        if msg is not None:
            actions.append(msg)

    return actions


def _is_improvement(val: float, prev_best: float | None, lower_better: bool) -> bool:
    """True when ``val`` beats ``prev_best`` (always true without a previous best)."""
    if prev_best is None:
        return True
    if lower_better:
        return val < prev_best
    return val > prev_best


@dataclass(frozen=True)
class ExportSource:
    """What :meth:`CheckpointLifecycleManager.export_tri_format` exports.

    The YOLO trainer replaces the model when ``is_yolo`` is set.
    """

    model: nn.Module | Any | None = None
    dummy_input: torch.Tensor | None = None
    is_yolo: bool = False
    yolo_trainer: Any | None = None

    def target(self) -> Any:
        """The object handed to the exporter."""
        return self.yolo_trainer if self.is_yolo else self.model


@dataclass(frozen=True)
class LifecyclePaths:
    """Authoritative per-model locations under ``LemGendaryModels/[ModelName]``."""

    model_hub_dir: Path
    checkpoints_dir: Path
    visualizations_dir: Path
    metrics_csv_path: Path
    progress_path: Path
    latest_path: Path
    best_path: Path

    @classmethod
    def for_hub(cls, model_hub_dir: Path) -> LifecyclePaths:
        """Resolve the checkpoint, visualization and metrics paths of one model hub dir."""
        checkpoints_dir = (model_hub_dir / "checkpoints").resolve()
        return cls(
            model_hub_dir=model_hub_dir,
            checkpoints_dir=checkpoints_dir,
            visualizations_dir=(model_hub_dir / "visualizations").resolve(),
            metrics_csv_path=(model_hub_dir / "metrics.csv").resolve(),
            progress_path=checkpoints_dir / "progress.pth",
            latest_path=checkpoints_dir / "latest.pth",
            best_path=checkpoints_dir / "best.pth",
        )


@dataclass
class LifecycleTracking:
    """Mutable save-cadence and best-score state of one lifecycle manager."""

    last_progress_save_time: float
    best_quality_score: float = float("-inf")
    vault_bests: dict[str, float] = field(default_factory=dict)


class CheckpointLifecycleManager:
    """Universal Checkpoint Lifecycle Manager enforcing strict governance rules."""

    def __init__(
        self,
        model_name: str,
        project_root: Path | None = None,
        sota_targets: dict[str, float] | None = None,
        min_progress_interval_sec: float = 900.0,  # 15 minutes
    ) -> None:
        self.model_name = model_name
        self.root = (project_root or get_project_root()).resolve()
        self.sota_targets = sota_targets or {}
        self.min_progress_interval_sec = min_progress_interval_sec

        # Authoritative paths under LemGendaryModels/[ModelName]
        self.paths = LifecyclePaths.for_hub(get_model_hub_dir(model_name, self.root))
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.visualizations_dir.mkdir(parents=True, exist_ok=True)

        # Enforce canonical naming: rename legacy checkpoints if canonical missing, purge redundant
        # duplicates
        canonicalize_checkpoints(self.paths.model_hub_dir, self.paths.checkpoints_dir)

        # Automated safeguard: Purge stale progress.pth if latest.pth represents an equal or newer
        # epoch
        if self.paths.progress_path.exists() and self.paths.latest_path.exists():
            try:
                p_meta = safe_load_checkpoint(self.paths.progress_path, map_location="cpu")
                l_meta = safe_load_checkpoint(self.paths.latest_path, map_location="cpu")
                p_ep = int(p_meta.get("epoch", 0)) if isinstance(p_meta, dict) else 0
                l_ep = int(l_meta.get("epoch", 0)) if isinstance(l_meta, dict) else 0
                if l_ep >= p_ep:
                    self.paths.progress_path.unlink()
                    logger.info(
                        "Purged obsolete progress checkpoint %s (progress ep %d <= latest ep %d)",
                        self.paths.progress_path,
                        p_ep,
                        l_ep,
                    )
            except CHECKPOINT_LOAD_ERRORS as clean_err:
                logger.debug("Progress checkpoint clean check notice: %s", clean_err)

        # Tracking state
        self.tracking = LifecycleTracking(last_progress_save_time=time.time())

        # Initialize vault tracking from sota_targets or existing metrics.csv
        self._initialize_vault_history()
        if self.paths.best_path.exists():
            try:
                best_data = safe_load_checkpoint(self.paths.best_path)
                if isinstance(best_data, dict):
                    saved_q = best_data.get("best_quality_score") or best_data.get("quality_score")
                    if saved_q is not None:
                        q_val = float(saved_q)
                        self.tracking.best_quality_score = max(
                            self.tracking.best_quality_score, q_val
                        )
            except (*CHECKPOINT_LOAD_ERRORS, TypeError) as e:
                logger.debug("Failed reading best.pth quality score: %s", e)

        # Pre-flight asset generation (README and cloud notebooks before Epoch 1)
        self.run_preflight(force=False)

    @property
    def model_hub_dir(self) -> Path:
        """``paths.model_hub_dir`` (read-only)."""
        return self.paths.model_hub_dir

    @property
    def checkpoints_dir(self) -> Path:
        """``paths.checkpoints_dir`` (read-only)."""
        return self.paths.checkpoints_dir

    @property
    def metrics_csv_path(self) -> Path:
        """``paths.metrics_csv_path`` (read-only)."""
        return self.paths.metrics_csv_path

    @property
    def progress_path(self) -> Path:
        """``paths.progress_path`` (read-only)."""
        return self.paths.progress_path

    @property
    def latest_path(self) -> Path:
        """``paths.latest_path`` (read-only)."""
        return self.paths.latest_path

    @property
    def best_path(self) -> Path:
        """``paths.best_path`` (read-only)."""
        return self.paths.best_path

    def run_preflight(self, force: bool = False) -> dict[str, Path]:
        """Generate pre-flight README docs and cloud training notebooks (Rank 0 only)."""
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return {}
        try:
            assets = generate_preflight_assets(self.model_name, self.root, force=force)
            logger.info(
                "Pre-flight assets verified for %s: %s", self.model_name, list(assets.keys())
            )
            return assets
        except PREFLIGHT_ERRORS as exc:
            logger.warning("Pre-flight generation notice for %s: %s", self.model_name, exc)
            return {}

    def _initialize_vault_history(self) -> None:
        """Scan existing metrics.csv to establish historical baselines for vault targets."""
        if not self.paths.metrics_csv_path.exists():
            return

        try:
            with open(self.paths.metrics_csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._absorb_quality_row(row)
                    self._absorb_vault_row(row)
        except OSError as exc:
            logger.debug("Failed parsing metrics.csv history for vault baselines: %s", exc)

    def _absorb_quality_row(self, row: dict[str, str]) -> None:
        """Raise the quality score baseline from one metrics.csv row."""
        for q_key in ["Quality", "Quality_Score", "quality_score"]:
            if row.get(q_key):
                try:
                    q_val = float(row[q_key])
                    self.tracking.best_quality_score = max(self.tracking.best_quality_score, q_val)
                except ValueError as q_err:
                    logger.debug(
                        "Skipping non-numeric %s value %r in metrics.csv: %s",
                        q_key,
                        row[q_key],
                        q_err,
                    )

    def _absorb_vault_row(self, row: dict[str, str]) -> None:
        """Update the SOTA target metric baselines from one metrics.csv row."""
        for target_key in self.sota_targets:
            lower_better = is_lower_better(target_key)
            norm_target = normalize_metric_name(target_key)
            for k, v in row.items():
                if not (k and normalize_metric_name(k) == norm_target and v):
                    continue
                try:
                    val = float(v)
                except ValueError as v_err:
                    logger.debug("Skipping non-numeric %s value %r in metrics.csv: %s", k, v, v_err)
                    continue
                if _is_improvement(val, self.tracking.vault_bests.get(target_key), lower_better):
                    self.tracking.vault_bests[target_key] = val

    def check_and_save_progress(
        self,
        epoch: int,
        phase: str,
        step: int,
        total_steps: int,
        payload_builder: Callable[[], dict[str, Any]],
    ) -> bool:
        """Evaluate and conditionally persist progress.pth checkpoint.

        Guardrails:
        1. At least min_progress_interval_sec (15 min) elapsed since last progress save.
        2. Progress within current phase is bounded:
           0.02 <= (step / total_steps) <= 0.98.
        Avoids saving during initial warmup (<2%) or immediately before epoch completion (>98%).
        """
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        if total_steps <= 0:
            return False

        progress_fraction = step / float(total_steps)

        # Progress checkpoint bounds enforcement: save across 2% to 98% of phase
        # Avoids saving during initial warmup (<2%) or immediately before latest.pth (>98%)
        if progress_fraction < 0.02 or progress_fraction > 0.98:
            return False

        now = time.time()
        elapsed = now - self.tracking.last_progress_save_time
        if elapsed < self.min_progress_interval_sec:
            return False

        # Build payload and write atomically
        payload = payload_builder()
        payload["progress_metadata"] = {
            "epoch": epoch,
            "phase": phase,
            "step": step,
            "total_steps": total_steps,
            "progress_fraction": round(progress_fraction, 4),
            "timestamp": now,
        }

        success = safe_atomic_save(payload, self.paths.progress_path)
        if success:
            self.tracking.last_progress_save_time = now
            logger.info(
                "Saved intra-epoch progress checkpoint to %s (Epoch %d, %s phase, step %d/%d = "
                "%.1f%%)",
                self.paths.progress_path,
                epoch,
                phase,
                step,
                total_steps,
                progress_fraction * 100.0,
            )
            print(
                f"[PROGRESS] Intra-epoch checkpoint captured: Epoch {epoch} ({phase} "
                f"{progress_fraction * 100.0:.1f}%) "
                f"to {self.paths.progress_path.name}",
                flush=True,
            )
        else:
            # Back off for 60 seconds on failure so subsequent 1-second batch iterations do not
            # hammer locked files
            self.tracking.last_progress_save_time = now - self.min_progress_interval_sec + 60.0
        return success

    def save_latest(self, epoch: int, payload: dict[str, Any]) -> bool:
        """Save latest.pth at epoch completion and immediately purge progress.pth (Rank 0 only)."""
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return True
        success = safe_atomic_save(payload, self.paths.latest_path)
        if not success:
            logger.error("Failed saving latest checkpoint at epoch %d", epoch)
            return False

        logger.info("Saved latest checkpoint to %s (Epoch %d)", self.paths.latest_path, epoch)

        # Immediate action: purge progress.pth upon saving latest.pth
        if self.paths.progress_path.exists():
            try:
                self.paths.progress_path.unlink()
                logger.info(
                    "Purged intra-epoch progress checkpoint %s after saving latest.pth",
                    self.paths.progress_path,
                )
            except OSError as unlink_err:
                logger.debug("Failed unlinking progress checkpoint: %s", unlink_err)

        # Reset 15-minute timer for upcoming epoch
        self.tracking.last_progress_save_time = time.time()
        return True

    def save_best(
        self,
        epoch: int,
        quality_score: float,
        payload: dict[str, Any],
        *,
        export: ExportSource | None = None,
    ) -> bool:
        """Save best.pth, invoke universal tri-format export, and refresh README (Rank 0 only)."""
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        if quality_score <= self.tracking.best_quality_score:
            return False

        self.tracking.best_quality_score = quality_score
        payload["best_quality_score"] = quality_score
        payload["best_epoch"] = epoch

        success = safe_atomic_save(payload, self.paths.best_path)
        if not success:
            logger.error("Failed saving best checkpoint at epoch %d", epoch)
            return False

        logger.info(
            "Saved new best checkpoint to %s (Epoch %d, Quality Score: %.4f)",
            self.paths.best_path,
            epoch,
            quality_score,
        )
        print(
            f"[SOTA] New best checkpoint achieved at Epoch {epoch} (Quality Score: "
            f"{quality_score:.4f}). "
            f"Persisted to {self.paths.best_path.name}",
            flush=True,
        )

        # Trigger Universal Tri-Format Export
        self.export_tri_format(export)

        # Refresh LemGendaryModels/[ModelName]/README.md
        self.refresh_readme(epoch, payload.get("best_metrics", {}))

        return True

    def check_and_save_vault(
        self,
        epoch: int,
        metrics: dict[str, float],
        payload: dict[str, Any],
    ) -> list[str]:
        """Check all target metrics against historical vault milestones.

        For each metric defined in sota_targets, if the current epoch sets a new
        best score, saves vault_[metric].pth.
        Returns list of newly achieved vault metric names.
        """
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return []
        achieved_vaults: list[str] = []

        for target_key in self.sota_targets:
            val = _vault_metric_value(target_key, metrics)
            if val is None:
                continue

            prev_best = self.tracking.vault_bests.get(target_key)
            if _is_improvement(val, prev_best, is_lower_better(target_key)):
                self.tracking.vault_bests[target_key] = val
                achieved_vaults.append(target_key)
                self._save_vault(target_key, val, epoch, payload)

        return achieved_vaults

    def _save_vault(self, target_key: str, val: float, epoch: int, payload: dict[str, Any]) -> None:
        """Write vault_[metric].pth for a new milestone."""
        vault_metric_clean = target_key.lower().strip().replace("-", "_")
        vault_filename = f"vault_{vault_metric_clean}.pth"
        vault_path = self.paths.checkpoints_dir / vault_filename
        vault_payload = dict(payload)
        vault_payload["vault_metric"] = target_key
        vault_payload["vault_score"] = val
        vault_payload["vault_epoch"] = epoch

        saved = safe_atomic_save(vault_payload, vault_path)
        if saved:
            logger.info(
                "Saved vault milestone checkpoint to %s: %s = %.4f (Epoch %d)",
                vault_path,
                target_key,
                val,
                epoch,
            )
            print(
                f"[VAULT] Milestone achieved: {target_key} = {val:.4f} at Epoch {epoch}. "
                f"Saved to {vault_filename}",
                flush=True,
            )

    def export_tri_format(self, source: ExportSource | None = None) -> dict[str, Path]:
        """Trigger universal tri-format export to LemGendaryModels/[ModelName]/."""
        print(
            f"[EXPORT] Initiating universal tri-format export for {self.model_name}...", flush=True
        )
        try:
            source = source or ExportSource()
            exported, _ = export_universal_model(
                model_key=self.model_name,
                model=source.target(),
                output_dir=self.paths.model_hub_dir,
                checkpoint_path=self.paths.best_path if self.paths.best_path.exists() else None,
                dummy_input=source.dummy_input,
                stage_dist=True,
            )
            return exported
        except EXPORT_ERRORS as exc:
            logger.error("Universal tri-format export failed for %s: %s", self.model_name, exc)
            return {}

    def refresh_readme(self, epoch: int, best_metrics: dict[str, float]) -> None:
        """Update LemGendaryModels/[ModelName]/README.md with current best metrics."""
        readme_path = self.paths.model_hub_dir / "README.md"
        try:
            config_path = self.root / "config.yaml"
            unified_dict = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                rel_u = cfg.get("unified_models", "unified_models_v2.yaml")
                u_path = self.root / rel_u
                if u_path.exists():
                    with open(u_path, encoding="utf-8") as uf:
                        unified_dict = yaml.safe_load(uf) or {}

            readme_content = build_model_readme(
                model_key=self.model_name,
                unified_models=unified_dict,
                epochs_trained=epoch,
                metrics=best_metrics,
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            logger.info("Refreshed model hub README at %s", readme_path)
        except (OSError, yaml.YAMLError, KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("Non-fatal README refresh notice for %s: %s", self.model_name, exc)


def _vault_metric_value(target_key: str, metrics: dict[str, float]) -> float | None:
    """First numeric metric whose normalized name matches ``target_key``, else None."""
    norm_target = normalize_metric_name(target_key)
    for k, v in metrics.items():
        if normalize_metric_name(k) == norm_target:
            try:
                return float(v)
            except (ValueError, TypeError) as conv_err:
                logger.debug(
                    "Ignoring non-numeric metric %s=%r for vault %s: %s",
                    k,
                    v,
                    target_key,
                    conv_err,
                )
    return None
