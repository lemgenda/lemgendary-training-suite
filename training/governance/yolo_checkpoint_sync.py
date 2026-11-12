"""Checkpoint, metrics and plot mirroring for the YOLOv8n curriculum governor.

Every artifact a rung produces is mirrored into the LemGendaryModels hub
(``LemGendaryModels/yolov8n``): best and latest weights plus the map50 / map50_95 vault
milestones under ``checkpoints/``, ``metrics.csv``, the Ultralytics validation plots and
the curriculum state JSON. On Kaggle with auto-sync enabled, a cloud sync is triggered.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.cloud.sync_policy import KaggleSyncPolicy
from training.cloud_sync import trigger_cloud_sync
from training.telemetry import TelemetryEngine
from training.utils.error_types import CLOUD_SYNC_START_ERRORS

logger = logging.getLogger("lemtrain.governance.yolo")

_PLOT_NAMES = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "results.png",
)


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


@dataclass
class YoloBest:
    """Best metrics so far, whether SOTA was reached and the best scores seen and exported."""

    best_overall_metrics: dict[str, float] = field(
        default_factory=lambda: {"map50": 0.0, "map50_95": 0.0}
    )
    sota_achieved: bool = False
    best_sota_quality: float = 0.0
    best_export_quality: float = 0.0


@dataclass(eq=False)
class YoloHubSync:
    """Mirrors rung artifacts into the models hub for ``YOLOCurriculumGovernor``.

    ``args`` and ``config`` decide the Kaggle cloud sync; ``best`` is the governor's
    best-so-far state recorded in the curriculum state JSON.
    """

    args: argparse.Namespace
    config: dict[str, Any]
    models_hub_dir: Path
    models_hub_ckpt_dir: Path
    runs_dir: Path
    telemetry: TelemetryEngine
    best: YoloBest = field(default_factory=YoloBest)

    def synchronize_checkpoints(
        self,
        stage_dir: Path,
        resolution: int,
        fraction: float = 1.0,
        stage_index: int = 1,
        global_epoch: int | None = None,
    ) -> None:
        """Mirror stage checkpoints to canonical LemGendaryModels/yolov8n/checkpoints/.

        All checkpoint artifacts are written exclusively to the LemGendaryModels hub.
        Enforces strict lifecycle rules: latest.pth purges progress.pth; vault milestones
        are persisted for map50 and map50_95.
        """
        frac_pct = round(fraction * 100)
        rung_dirs = [
            stage_dir / f"rung_{resolution}_f{frac_pct}",
            stage_dir / f"rung_{resolution}",
            stage_dir,
        ]
        weights_dirs = [d / "weights" for d in rung_dirs]
        stage_weights_dir = next((d for d in weights_dirs if d.exists()), weights_dirs[0])

        hub_ckpt_dir = self.models_hub_ckpt_dir
        hub_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._sync_best_weights(stage_weights_dir / "best.pt", hub_ckpt_dir)
        self._sync_latest_weights(stage_weights_dir / "last.pt", hub_ckpt_dir)

        # Synchronize metrics CSV to LemGendaryModels/yolov8n/
        csv_source = Path(self.telemetry.metrics_csv_path)
        if csv_source.exists():
            try:
                shutil.copy2(csv_source, self.models_hub_dir / "metrics.csv")
            except OSError as csv_err:
                logger.debug("Non-fatal metrics CSV sync error: %s", csv_err)

        rung_dir = next((d for d in rung_dirs if d.exists()), None)
        if rung_dir is not None:
            self._sync_plots(rung_dir)

        if global_epoch is not None:
            self._save_curriculum_state(
                hub_ckpt_dir,
                {
                    "model_key": "yolov8n",
                    "global_epoch": global_epoch,
                    "resolution": resolution,
                    "fraction": round(fraction, 2),
                    "stage_index": stage_index,
                    "best_metrics": self.best.best_overall_metrics,
                    "sota_achieved": self.best.sota_achieved,
                },
            )
            self._trigger_remote_sync(global_epoch)

    def _sync_best_weights(self, best_cand: Path, hub_ckpt_dir: Path) -> None:
        """Copy best weights to the hub, plus the vault milestones once metrics exist."""
        if not _nonempty(best_cand):
            return
        try:
            shutil.copy2(best_cand, hub_ckpt_dir / "best.pt")
            shutil.copy2(best_cand, hub_ckpt_dir / "best.pth")
            # Vault milestone checkpoints for YOLO targets
            if self.best.best_overall_metrics["map50"] > 0:
                shutil.copy2(best_cand, hub_ckpt_dir / "vault_map50.pth")
            if self.best.best_overall_metrics["map50_95"] > 0:
                shutil.copy2(best_cand, hub_ckpt_dir / "vault_map50_95.pth")
        except OSError as copy_err:
            logger.debug("Non-fatal checkpoint copy error for best weights: %s", copy_err)

    @staticmethod
    def _sync_latest_weights(last_cand: Path, hub_ckpt_dir: Path) -> None:
        """Copy latest weights to the hub and purge the superseded progress.pth."""
        if not _nonempty(last_cand):
            return
        try:
            shutil.copy2(last_cand, hub_ckpt_dir / "latest.pth")
            shutil.copy2(last_cand, hub_ckpt_dir / "last.pt")
            # Immediate action: purge progress.pth upon saving latest
            progress_file = hub_ckpt_dir / "progress.pth"
            if progress_file.exists():
                progress_file.unlink()
        except OSError as copy_err:
            logger.debug("Non-fatal checkpoint copy error for latest weights: %s", copy_err)

    def _sync_plots(self, rung_dir: Path) -> None:
        """Copy validation plots (confusion matrices, PR/F1 curves) to the hub."""
        for plot_name in _PLOT_NAMES:
            plot_file = rung_dir / plot_name
            if _nonempty(plot_file):
                try:
                    vis_dir = self.models_hub_dir / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(plot_file, vis_dir / plot_name)
                except OSError as plot_err:
                    logger.debug("Non-fatal plot sync error: %s", plot_err)

    @staticmethod
    def _save_curriculum_state(hub_ckpt_dir: Path, curriculum_state: dict[str, Any]) -> None:
        try:
            with open(hub_ckpt_dir / "curriculum_state.json", "w", encoding="utf-8") as f:
                json.dump(curriculum_state, f, indent=2)
        except OSError as state_err:
            logger.debug("Non-fatal curriculum state save error: %s", state_err)

    def _trigger_remote_sync(self, global_epoch: int) -> None:
        """Automated cloud sync trigger when running on Kaggle with auto-sync enabled."""
        if (
            getattr(self.args, "auto_sync", False)
            and getattr(self.args, "env", "local") == "kaggle"
            and KaggleSyncPolicy.from_config(self.config).should_sync(global_epoch, False)
        ):
            try:
                trigger_cloud_sync("yolov8n", global_epoch, self.config)
            except CLOUD_SYNC_START_ERRORS as sync_err:
                logger.warning(
                    "Non-fatal cloud sync trigger error at epoch %s: %s", global_epoch, sync_err
                )
