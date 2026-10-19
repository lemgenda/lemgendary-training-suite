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
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from training.checkpoint.manager import safe_atomic_save
from training.export.universal_exporter import (
    export_tri_format_pytorch,
    export_tri_format_yolo,
    export_universal_model,
    get_model_hub_dir,
)
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.checkpoint.lifecycle")


LOWER_IS_BETTER_METRICS: set[str] = {
    "loss",
    "val_loss",
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
    cleaned = str(name).strip().lower().replace("-", "_")
    unscored = cleaned.replace("_", "")
    return CANONICAL_METRIC_ALIASES.get(cleaned, CANONICAL_METRIC_ALIASES.get(unscored, cleaned))


def is_lower_better(metric_name: str) -> bool:
    """Determine whether a metric improves when its value decreases."""
    norm = normalize_metric_name(metric_name)
    if norm in LOWER_IS_BETTER_METRICS:
        return True
    return any(sub in norm for sub in ("loss", "mae", "mse", "drawdown", "lpips", "fid"))


def canonicalize_checkpoints(model_hub_dir: Path, checkpoints_dir: Path) -> list[str]:
    """Scan model directory for legacy-named checkpoints and rename to canonical if canonical does not exist.

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
        if not pth.is_file():
            continue

        name_lower = pth.name.lower()

        # Determine target canonical filename
        target_name: str | None = None
        if "vault_" in name_lower:
            metric_part = name_lower.split("vault_")[-1].replace(".pth", "")
            metric_clean = normalize_metric_name(metric_part)
            target_name = f"vault_{metric_clean}.pth"
        elif "best" in name_lower:
            target_name = "best.pth"
        elif "latest" in name_lower or "last" in name_lower:
            target_name = "latest.pth"
        elif "progress" in name_lower:
            target_name = "progress.pth"

        if not target_name:
            continue

        target_path = checkpoints_dir / target_name

        if pth.resolve() == target_path.resolve():
            continue

        if not target_path.exists():
            try:
                shutil.move(str(pth), str(target_path))
                msg = f"Renamed legacy checkpoint {pth.name} -> {target_path.name}"
                actions.append(msg)
                logger.info(msg)
            except OSError as move_err:
                logger.warning("Failed renaming checkpoint %s -> %s: %s", pth, target_path, move_err)
        else:
            try:
                pth.unlink()
                msg = f"Purged redundant duplicate checkpoint {pth.name} (canonical {target_path.name} exists)"
                actions.append(msg)
                logger.info(msg)
            except OSError as unlink_err:
                logger.debug("Failed unlinking duplicate checkpoint %s: %s", pth, unlink_err)

    return actions


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
        self.model_hub_dir = get_model_hub_dir(model_name, self.root)
        self.checkpoints_dir = (self.model_hub_dir / "checkpoints").resolve()
        self.visualizations_dir = (self.model_hub_dir / "visualizations").resolve()
        self.metrics_csv_path = (self.model_hub_dir / "metrics.csv").resolve()

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

        # Enforce canonical naming: rename legacy checkpoints if canonical missing, purge redundant duplicates
        canonicalize_checkpoints(self.model_hub_dir, self.checkpoints_dir)

        self.progress_path = self.checkpoints_dir / "progress.pth"
        self.latest_path = self.checkpoints_dir / "latest.pth"
        self.best_path = self.checkpoints_dir / "best.pth"

        # Tracking state
        self.last_progress_save_time = time.time()
        self.best_quality_score: float = float("-inf")
        self.vault_bests: dict[str, float] = {}

        # Initialize vault tracking from sota_targets or existing metrics.csv
        self._initialize_vault_history()

        # Pre-flight asset generation (README and cloud notebooks before Epoch 1)
        self.run_preflight(force=False)

    def run_preflight(self, force: bool = False) -> dict[str, Path]:
        """Generate pre-flight deployment documentation and cloud training notebooks."""
        try:
            from training.checkpoint.preflight_generator import generate_preflight_assets

            assets = generate_preflight_assets(self.model_name, self.root, force=force)
            logger.info("Pre-flight assets verified for %s: %s", self.model_name, list(assets.keys()))
            return assets
        except Exception as exc:
            logger.warning("Pre-flight generation notice for %s: %s", self.model_name, exc)
            return {}

    def _initialize_vault_history(self) -> None:
        """Scan existing metrics.csv to establish historical baselines for vault targets."""
        if not self.metrics_csv_path.exists():
            return

        try:
            with open(self.metrics_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Quality score baseline
                    for q_key in ["Quality", "Quality_Score", "quality_score"]:
                        if row.get(q_key):
                            try:
                                q_val = float(row[q_key])
                                if q_val > self.best_quality_score:
                                    self.best_quality_score = q_val
                            except ValueError:
                                pass

                    # SOTA target metric baselines
                    for target_key in self.sota_targets:
                        lower_better = is_lower_better(target_key)
                        norm_target = normalize_metric_name(target_key)
                        for k, v in row.items():
                            if k and normalize_metric_name(k) == norm_target and v:
                                try:
                                    val = float(v)
                                    cur = self.vault_bests.get(target_key)
                                    if cur is None:
                                        self.vault_bests[target_key] = val
                                    elif lower_better:
                                        if val < cur:
                                            self.vault_bests[target_key] = val
                                    else:
                                        if val > cur:
                                            self.vault_bests[target_key] = val
                                except ValueError:
                                    pass
        except OSError as exc:
            logger.debug("Failed parsing metrics.csv history for vault baselines: %s", exc)

    def check_and_save_progress(
        self,
        epoch: int,
        phase: str,
        step: int,
        total_steps: int,
        payload_builder: Callable[[], dict[str, Any]],
    ) -> bool:
        """Evaluate and conditionally persist progress.pth checkpoint.

        Strict guardrails:
        1. At least min_progress_interval_sec (15 min) elapsed since last progress save.
        2. Progress within current phase is strictly bounded between 5% and 50%:
           0.05 <= (step / total_steps) <= 0.50.
        Never saved below 5% or above 50% regardless of elapsed time.
        """
        if total_steps <= 0:
            return False

        progress_fraction = step / float(total_steps)

        # Strict bounds enforcement: [5%, 50%]
        if progress_fraction < 0.05 or progress_fraction > 0.50:
            return False

        now = time.time()
        elapsed = now - self.last_progress_save_time
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

        success = safe_atomic_save(payload, self.progress_path)
        if success:
            self.last_progress_save_time = now
            logger.info(
                "Saved intra-epoch progress checkpoint to %s (Epoch %d, %s phase, step %d/%d = %.1f%%)",
                self.progress_path,
                epoch,
                phase,
                step,
                total_steps,
                progress_fraction * 100.0,
            )
            print(
                f"[PROGRESS] Intra-epoch checkpoint captured: Epoch {epoch} ({phase} {progress_fraction * 100.0:.1f}%) "
                f"to {self.progress_path.name}",
                flush=True,
            )
        else:
            # Back off for 60 seconds on failure so subsequent 1-second batch iterations do not hammer locked files
            self.last_progress_save_time = now - self.min_progress_interval_sec + 60.0
        return success

    def save_latest(self, epoch: int, payload: dict[str, Any]) -> bool:
        """Save latest.pth at epoch completion and immediately purge progress.pth."""
        success = safe_atomic_save(payload, self.latest_path)
        if not success:
            logger.error("Failed saving latest checkpoint at epoch %d", epoch)
            return False

        logger.info("Saved latest checkpoint to %s (Epoch %d)", self.latest_path, epoch)

        # Immediate action: purge progress.pth upon saving latest.pth
        if self.progress_path.exists():
            try:
                self.progress_path.unlink()
                logger.info("Purged intra-epoch progress checkpoint %s after saving latest.pth", self.progress_path)
            except OSError as unlink_err:
                logger.debug("Failed unlinking progress checkpoint: %s", unlink_err)

        # Reset 15-minute timer for upcoming epoch
        self.last_progress_save_time = time.time()
        return True

    def save_best(
        self,
        epoch: int,
        quality_score: float,
        payload: dict[str, Any],
        model: nn.Module | Any | None = None,
        dummy_input: torch.Tensor | None = None,
        is_yolo: bool = False,
        yolo_trainer: Any | None = None,
    ) -> bool:
        """Save best.pth, invoke universal tri-format export, and refresh README."""
        if quality_score <= self.best_quality_score:
            return False

        self.best_quality_score = quality_score
        payload["best_quality_score"] = quality_score
        payload["best_epoch"] = epoch

        success = safe_atomic_save(payload, self.best_path)
        if not success:
            logger.error("Failed saving best checkpoint at epoch %d", epoch)
            return False

        logger.info(
            "Saved new best checkpoint to %s (Epoch %d, Quality Score: %.4f)",
            self.best_path,
            epoch,
            quality_score,
        )
        print(
            f"[SOTA] New best checkpoint achieved at Epoch {epoch} (Quality Score: {quality_score:.4f}). "
            f"Persisted to {self.best_path.name}",
            flush=True,
        )

        # Trigger Universal Tri-Format Export
        self.export_tri_format(
            model=model,
            dummy_input=dummy_input,
            is_yolo=is_yolo,
            yolo_trainer=yolo_trainer,
        )

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
        achieved_vaults: list[str] = []

        for target_key in self.sota_targets:
            norm_target = normalize_metric_name(target_key)
            val: float | None = None
            for k, v in metrics.items():
                if normalize_metric_name(k) == norm_target:
                    try:
                        val = float(v)
                        break
                    except (ValueError, TypeError):
                        pass

            if val is None:
                continue

            lower_better = is_lower_better(target_key)
            prev_best = self.vault_bests.get(target_key)

            if prev_best is None:
                is_milestone = True
            elif lower_better:
                is_milestone = (val < prev_best)
            else:
                is_milestone = (val > prev_best)

            if is_milestone:
                self.vault_bests[target_key] = val
                achieved_vaults.append(target_key)

                vault_metric_clean = target_key.lower().strip().replace("-", "_")
                vault_filename = f"vault_{vault_metric_clean}.pth"
                vault_path = self.checkpoints_dir / vault_filename
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

        return achieved_vaults

    def export_tri_format(
        self,
        model: nn.Module | Any | None = None,
        dummy_input: torch.Tensor | None = None,
        is_yolo: bool = False,
        yolo_trainer: Any | None = None,
    ) -> dict[str, Path]:
        """Trigger universal tri-format export to LemGendaryModels/[ModelName]/."""
        print(f"[EXPORT] Initiating universal tri-format export for {self.model_name}...", flush=True)
        try:
            target_model = yolo_trainer if is_yolo else model
            exported, _ = export_universal_model(
                model_key=self.model_name,
                model=target_model,
                output_dir=self.model_hub_dir,
                checkpoint_path=self.best_path if self.best_path.exists() else None,
                dummy_input=dummy_input,
                stage_dist=True,
            )
            return exported
        except Exception as exc:
            logger.error("Universal tri-format export failed for %s: %s", self.model_name, exc)
            return {}

    def refresh_readme(self, epoch: int, best_metrics: dict[str, float]) -> None:
        """Update LemGendaryModels/[ModelName]/README.md with current best metrics."""
        readme_path = self.model_hub_dir / "README.md"
        try:
            import yaml

            from training.doc_generator import build_model_readme

            config_path = self.root / "config.yaml"
            unified_dict = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                rel_u = cfg.get("unified_models", "unified_models_v2.yaml")
                u_path = self.root / rel_u
                if u_path.exists():
                    with open(u_path, "r", encoding="utf-8") as uf:
                        unified_dict = yaml.safe_load(uf) or {}

            readme_content = build_model_readme(
                model_key=self.model_name,
                unified_models=unified_dict,
                epochs_trained=epoch,
                metrics=best_metrics,
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            logger.info("Refreshed model hub README at %s", readme_path)
        except Exception as exc:
            logger.debug("Non-fatal README refresh notice for %s: %s", self.model_name, exc)
