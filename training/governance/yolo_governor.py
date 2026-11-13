"""Governed YOLOv8n Training Orchestrator and Curriculum Governor.

Coordinates multi-stage spatial resolution ladders (320px -> 480px -> 640px),
hardware-aware Sawtooth VRAM batch allocation, dataset fraction scaling,
numerical AMP stabilization, and metric telemetry synchronization while
preserving native Ultralytics inner-loop optimizations.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from data.yolo_config_gen import generate_yolo_yaml
from training.export.base import EXPORT_ERRORS
from training.export.universal_exporter import export_tri_format_yolo
from training.governance.yolo_checkpoint_sync import YoloHubSync
from training.governance.yolo_ladder import (
    BatchBudget,
    YOLOLadderStage,
    build_ladder_curriculum,
    compute_safe_batch_size,
    is_amp_supported_for_device,
)
from training.governance.yolo_run import YoloModelSpec, YoloRunHooks, YoloRunMixin
from training.telemetry import TelemetryEngine
from training.training.summary import TrainingSummary
from training.utils.lazy_imports import load_ultralytics
from training.utils.paths import get_model_hub_dir

__all__ = [
    "BatchBudget",
    "YOLOCurriculumGovernor",
    "YOLOLadderStage",
    "build_ladder_curriculum",
    "compute_safe_batch_size",
    "create_yolo_model",
    "is_amp_supported_for_device",
    "run_governed_yolo_training",
]

logger = logging.getLogger("lemtrain.governance.yolo")


def create_yolo_model(weights: str | Path) -> Any:
    """Instantiate an Ultralytics YOLO model (single construction seam for the governor).

    ``ultralytics`` is a heavy optional runtime dependency, so it is imported lazily
    here and nowhere else in this module; tests substitute this factory instead of
    requiring the real package.
    """
    return load_ultralytics().YOLO(str(weights))


def _model_spec(model_info: dict[str, Any], config: dict[str, Any]) -> YoloModelSpec:
    """Merge the config's optimization block over the registry entry's."""
    merged_opt = dict(model_info.get("optimization", {}))
    if isinstance(config.get("optimization"), dict):
        merged_opt.update(config["optimization"])
    return YoloModelSpec(
        model_info=model_info,
        opt_config=merged_opt,
        sota_targets=model_info.get(
            "sota_targets",
            {"map50": 0.54, "map50_95": 0.39},
        ),
    )


def _migrate_legacy_runs(models_hub_dir: Path, runs_dir: Path) -> None:
    """Move a legacy ``runs`` directory out of the models hub into ``runs_dir``."""
    legacy_runs = models_hub_dir / "runs"
    if legacy_runs.exists():
        try:
            for item in legacy_runs.iterdir():
                dst_item = runs_dir / item.name
                if not dst_item.exists():
                    shutil.move(str(item), str(dst_item))
            shutil.rmtree(legacy_runs)
        except OSError as exc:
            logger.warning(
                "Legacy runs migration from %s to %s incomplete: %s",
                legacy_runs,
                runs_dir,
                exc,
            )


class YOLOCurriculumGovernor(YoloRunMixin):
    """Universal Governor orchestrator for YOLOv8n training execution."""

    def __init__(
        self,
        args: argparse.Namespace,
        config: dict[str, Any],
        project_root: Path,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        unified_models_rel = config.get("unified_models", "unified_models_v2.yaml")
        unified_models_path = project_root / unified_models_rel
        registry = {}
        if unified_models_path.exists():
            with open(unified_models_path, encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}

        self.hooks = YoloRunHooks(project_root, on_epoch_end, cancel_check)
        self.spec = _model_spec(registry.get("yolov8n", {}), config)

        # All training artifacts are stored exclusively under LemGendaryModels — the training
        # suite project root is for code only, never for generated model files.
        models_hub_dir = get_model_hub_dir("yolov8n", project_root)
        models_hub_ckpt_dir = models_hub_dir / "checkpoints"
        # Curriculum stage outputs go under local checkpoints directory to keep LemGendaryModels
        # pristine
        runs_dir = (project_root / "checkpoints" / "yolov8n" / "runs").resolve()

        models_hub_dir.mkdir(parents=True, exist_ok=True)
        models_hub_ckpt_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        _migrate_legacy_runs(models_hub_dir, runs_dir)

        # Primary metrics and telemetry stream directly from LemGendaryModels/yolov8n
        telemetry = TelemetryEngine(export_dir=str(models_hub_dir), task_type="yolo")
        telemetry.validate_and_initialize_csv()

        self.sync = YoloHubSync(
            args=args,
            config=config,
            models_hub_dir=models_hub_dir,
            models_hub_ckpt_dir=models_hub_ckpt_dir,
            runs_dir=runs_dir,
            telemetry=telemetry,
        )

    def _export_if_new_best(self, trainer: Any) -> None:
        """Export PyTorch and ONNX artifacts whenever a new global best checkpoint is saved.

        Runs from the Ultralytics on_model_save hook, which fires after best.pt has been
        written for the current epoch. A model is exported only when Ultralytics marks the
        epoch as its best and the combined quality score beats every prior export.
        """
        try:
            fitness = getattr(trainer, "fitness", None)
            best_fitness = getattr(trainer, "best_fitness", None)
            if fitness is None or best_fitness is None or float(fitness) != float(best_fitness):
                return

            raw_metrics = getattr(trainer, "metrics", {}) or {}
            map50 = float(raw_metrics.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(raw_metrics.get("metrics/mAP50-95(B)", 0.0))
            quality = (map50_95 * 0.7) + (map50 * 0.3)
            if quality <= self.best_export_quality:
                return

            best_path = getattr(trainer, "best", None)
            if not (best_path and Path(best_path).exists() and Path(best_path).stat().st_size > 0):
                return

            self.best_export_quality = quality
            target_map50 = self.sota_targets.get("map50", 0.54)
            target_map50_95 = self.sota_targets.get("map50_95", 0.39)
            sota_met = map50 >= target_map50 and map50_95 >= target_map50_95
            tag = "SOTA ATTAINED" if sota_met else "NEW BEST"
            print(
                f"\n[GOVERNOR] [{tag}] mAP50={map50:.4f}, mAP50-95={map50_95:.4f} "
                f"(quality {quality:.4f}). Exporting PyTorch and ONNX artifacts to "
                f"{self.models_hub_dir}...",
                flush=True,
            )
            self.export_sota_models(Path(best_path))
        except EXPORT_ERRORS as export_err:
            logger.warning("New-best export hook failed: %s", export_err)

    def export_sota_models(self, source_weights: Path) -> None:
        """Export high-performance SOTA model as tri-format ONNX and PyTorch artifacts.

        Exports production-ready deployment assets directly to LemGendaryModels/yolov8n/:
          - yolov8n.pt (native PyTorch FP32 weights)
          - yolov8n_FP32.onnx + yolov8n_FP32.onnx.data (FP32 ONNX graph with external sidecar)
          - yolov8n.onnx (optimized FP16 ONNX inference graph)
        """
        if not source_weights.exists() or source_weights.stat().st_size == 0:
            logger.warning(
                "Cannot export SOTA model: source weights %s missing or empty.", source_weights
            )
            return

        try:
            export_model = create_yolo_model(str(source_weights))
            export_tri_format_yolo(
                trainer=export_model,
                model_key="yolov8n",
                output_dir=self.models_hub_dir,
            )
            print(
                "[GOVERNOR] [SOTA EXPORT] Tri-format SOTA models exported to "
                f"{self.models_hub_dir}",
                flush=True,
            )
        except EXPORT_ERRORS as export_err:
            logger.warning("Error during tri-format SOTA model export: %s", export_err)

    def _create_model(self, weights: str | Path) -> Any:
        """Build a YOLO model through the module-level construction seam."""
        return create_yolo_model(weights)

    def _dataset_yaml(self) -> Any:
        """Generate the Ultralytics dataset YAML for every configured yolov8n dataset."""
        return generate_yolo_yaml(self.config, "yolov8n", {"yolov8n": self.model_info})


def run_governed_yolo_training(
    args: argparse.Namespace,
    config: dict[str, Any],
    project_root: Path,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TrainingSummary:
    """Entry point for governed YOLOv8n training dispatch."""
    governor = YOLOCurriculumGovernor(
        args=args,
        config=config,
        project_root=project_root,
        on_epoch_end=on_epoch_end,
        cancel_check=cancel_check,
    )
    return governor.run()
