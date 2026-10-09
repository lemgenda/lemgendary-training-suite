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
from pathlib import Path
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from training.checkpoint.manager import safe_atomic_save
from training.export.universal_exporter import (
    export_tri_format_pytorch,
    export_tri_format_yolo,
    get_model_hub_dir,
)
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.checkpoint.lifecycle")


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
        self.training_dir = (self.model_hub_dir / "training").resolve()
        self.metrics_csv_path = (self.model_hub_dir / "metrics.csv").resolve()

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.progress_path = self.checkpoints_dir / "progress.pth"
        self.latest_path = self.checkpoints_dir / "latest.pth"
        self.best_path = self.checkpoints_dir / "best.pth"

        # Tracking state
        self.last_progress_save_time = time.time()
        self.best_quality_score: float = float("-inf")
        self.vault_bests: dict[str, float] = {}

        # Initialize vault tracking from sota_targets or existing metrics.csv
        self._initialize_vault_history()

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
                        if q_key in row and row[q_key]:
                            try:
                                q_val = float(row[q_key])
                                if q_val > self.best_quality_score:
                                    self.best_quality_score = q_val
                            except ValueError:
                                pass

                    # SOTA target metric baselines
                    for target_key in self.sota_targets:
                        # Match case-insensitively with stripped names
                        for k, v in row.items():
                            if k and k.lower() == target_key.lower() and v:
                                try:
                                    val = float(v)
                                    cur = self.vault_bests.get(target_key, float("-inf"))
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
            # Locate metric in metrics dict case-insensitively
            val: float | None = None
            for k, v in metrics.items():
                if k.lower() == target_key.lower():
                    try:
                        val = float(v)
                        break
                    except (ValueError, TypeError):
                        pass

            if val is None:
                continue

            prev_best = self.vault_bests.get(target_key, float("-inf"))
            if val > prev_best:
                self.vault_bests[target_key] = val
                achieved_vaults.append(target_key)

                vault_filename = f"vault_{target_key.lower()}.pth"
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
            if is_yolo:
                trainer = yolo_trainer or model
                if trainer is not None:
                    return export_tri_format_yolo(
                        trainer=trainer,
                        model_key=self.model_name,
                        output_dir=self.model_hub_dir,
                    )
            elif model is not None and isinstance(model, nn.Module):
                # Fallback dummy input if not provided
                if dummy_input is None:
                    device = next(model.parameters()).device
                    dummy_input = torch.randn(1, 3, 256, 256, device=device)
                return export_tri_format_pytorch(
                    model=model,
                    model_key=self.model_name,
                    dummy_input=dummy_input,
                    output_dir=self.model_hub_dir,
                )
            else:
                logger.warning("No exportable model or trainer instance provided for %s", self.model_name)
        except Exception as exc:
            logger.error("Universal tri-format export failed for %s: %s", self.model_name, exc)

        return {}

    def refresh_readme(self, epoch: int, best_metrics: dict[str, float]) -> None:
        """Update LemGendaryModels/[ModelName]/README.md with current best metrics."""
        readme_path = self.model_hub_dir / "README.md"
        try:
            from training.doc_generator import build_model_readme
            import yaml

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
                final_epoch=epoch,
                best_metrics=best_metrics,
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            logger.info("Refreshed model hub README at %s", readme_path)
        except Exception as exc:
            logger.debug("Non-fatal README refresh notice for %s: %s", self.model_name, exc)
