"""Governed YOLOv8n Training Orchestrator and Curriculum Governor.

Coordinates multi-stage spatial resolution ladders (320px -> 480px -> 640px),
hardware-aware Sawtooth VRAM batch allocation, dataset fraction scaling,
numerical AMP stabilization, and metric telemetry synchronization while
preserving native Ultralytics inner-loop optimizations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml

from data.yolo_config_gen import generate_yolo_yaml
from training.telemetry import TelemetryEngine
from training.training.engine import TrainingSummary

logger = logging.getLogger("lemtrain.governance.yolo")


@dataclass
class YOLOLadderStage:
    """Configuration for an individual curriculum resolution ladder stage."""

    stage_index: int
    resolution: int
    batch_size: int
    fraction: float
    target_epochs: int
    patience: int


def is_amp_supported_for_device(device_arg: str) -> bool:
    """Evaluate whether AMP FP16 is numerically stable on target GPU hardware.

    NVIDIA GTX 16xx series cards (e.g. GTX 1650, 1660, 1660 Ti with TU117/TU116 chips)
    lack Tensor Cores and have documented cuDNN FP16 gradient underflow issues that
    produce NaN losses during YOLO backward passes. In such environments, FP32
    is required for stability.
    """
    if not torch.cuda.is_available():
        return False

    try:
        if device_arg in ("cpu", "-1"):
            return False

        device_index = 0
        if device_arg.isdigit():
            device_index = int(device_arg)
        elif "," in device_arg:
            first_dev = device_arg.split(",")[0].strip()
            if first_dev.isdigit():
                device_index = int(first_dev)

        dev_name = torch.cuda.get_device_name(device_index).lower()
        if any(unsupported in dev_name for unsupported in ["gtx 1650", "gtx 1660", "gtx 10", "gtx 9"]):
            return False

        major, minor = torch.cuda.get_device_capability(device_index)
        if major == 7 and minor == 5 and "rtx" not in dev_name:
            return False

        return True
    except (RuntimeError, ValueError) as exc:
        logger.debug("Failed to probe GPU device capability for AMP: %s", exc)
        return False


def compute_safe_batch_size(
    imgsz: int,
    vram_gb: float,
    requested_batch: Optional[int] = None,
) -> int:
    """Calculate maximum safe physical batch size under Sawtooth VRAM guard.

    Prevents CUDA Out-Of-Memory crashes at high spatial resolutions by
    dynamically pacing minibatch allocation based on available GPU VRAM.
    """
    if vram_gb <= 4.5:
        ladder_limits = {320: 16, 480: 8, 640: 4, 1024: 2}
    elif vram_gb <= 8.5:
        ladder_limits = {320: 32, 480: 16, 640: 8, 1024: 4}
    elif vram_gb <= 16.5:
        ladder_limits = {320: 64, 480: 32, 640: 16, 1024: 8}
    else:
        ladder_limits = {320: 128, 480: 64, 640: 32, 1024: 16}

    max_safe = 4
    for res_bound, safe_limit in sorted(ladder_limits.items()):
        if imgsz <= res_bound:
            max_safe = safe_limit
            break
    else:
        max_safe = 2

    if requested_batch is not None and requested_batch > 0:
        return min(requested_batch, max_safe)

    return max_safe


def build_ladder_curriculum(
    res_ladder: List[int],
    total_epochs: int,
    vram_gb: float,
    requested_batch: Optional[int],
    opt_config: Dict[str, Any],
    start_resolution: Optional[int] = None,
) -> List[YOLOLadderStage]:
    """Construct deterministic multi-stage resolution ladder curriculum."""
    rungs = sorted(list(set(res_ladder))) if res_ladder else [320, 480, 640]

    if start_resolution is not None and start_resolution in rungs:
        rungs = [r for r in rungs if r >= start_resolution]
        if not rungs:
            rungs = [start_resolution]

    num_stages = len(rungs)
    if num_stages == 1:
        epoch_allocations = [total_epochs]
    elif num_stages == 2:
        s1 = max(1, int(total_epochs * 0.40))
        s2 = max(1, total_epochs - s1)
        epoch_allocations = [s1, s2]
    else:
        s1 = max(1, int(total_epochs * 0.25))
        s2 = max(1, int(total_epochs * 0.35))
        s3 = max(1, total_epochs - s1 - s2)
        epoch_allocations = [s1, s2, s3]

    initial_frac = float(opt_config.get("initial_fraction", 0.3))
    mid_frac = float(opt_config.get("high_fidelity_fraction", 0.6))
    if num_stages == 1:
        fractions = [1.0]
    elif num_stages == 2:
        fractions = [initial_frac, 1.0]
    else:
        fractions = [initial_frac, mid_frac, 1.0]

    stages: List[YOLOLadderStage] = []
    patience_base = int(opt_config.get("plateau_patience", 10))

    for idx, (res, ep_budget, frac) in enumerate(zip(rungs, epoch_allocations, fractions)):
        safe_batch = compute_safe_batch_size(
            imgsz=res,
            vram_gb=vram_gb,
            requested_batch=requested_batch,
        )
        stage_patience = max(5, patience_base + (idx * 2))
        stages.append(
            YOLOLadderStage(
                stage_index=idx + 1,
                resolution=res,
                batch_size=safe_batch,
                fraction=frac,
                target_epochs=ep_budget,
                patience=stage_patience,
            )
        )

    return stages


class YOLOCurriculumGovernor:
    """Universal Governor orchestrator for YOLOv8n training execution."""

    def __init__(
        self,
        args: argparse.Namespace,
        config: Dict[str, Any],
        project_root: Path,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> None:
        self.args = args
        self.config = config
        self.project_root = project_root
        self.on_epoch_end = on_epoch_end

        unified_models_rel = config.get("unified_models", "unified_models_v2.yaml")
        unified_models_path = project_root / unified_models_rel
        with open(unified_models_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        self.model_info: Dict[str, Any] = registry.get("yolov8n", {})
        self.opt_config: Dict[str, Any] = self.model_info.get("optimization", {})
        self.sota_targets: Dict[str, float] = self.model_info.get(
            "sota_targets",
            {"map50": 0.54, "map50_95": 0.39},
        )

        self.export_dir = project_root / "export" / "yolov8n"
        self.checkpoints_dir = project_root / "checkpoints" / "yolov8n"
        self.models_hub_dir = (project_root / ".." / "LemGendaryModels" / "yolov8n").resolve()

        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_hub_dir.mkdir(parents=True, exist_ok=True)

        self.telemetry = TelemetryEngine(export_dir=str(self.checkpoints_dir), task_type="yolo")
        self.telemetry.validate_and_initialize_csv()

    def _resolve_devices(self) -> str:
        """Resolve CUDA device configuration string."""
        if not torch.cuda.is_available():
            return "cpu"
        count = torch.cuda.device_count()
        if count >= 2:
            return ",".join(str(i) for i in range(count))
        return "0"

    def _resolve_vram_gb(self) -> float:
        """Query primary GPU total physical VRAM capacity in gigabytes."""
        if torch.cuda.is_available():
            try:
                return float(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            except (RuntimeError, ValueError) as exc:
                logger.debug("Failed to read GPU memory properties: %s", exc)
                return 4.0
        return 8.0

    def run(self) -> TrainingSummary:
        """Execute full curriculum-governed multi-stage training pipeline."""
        start_time = time.time()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for yolov8n training. Install it with: pip install ultralytics"
            ) from exc

        device_arg = self._resolve_devices()
        vram_gb = self._resolve_vram_gb()
        is_amp_safe = is_amp_supported_for_device(device_arg)

        yolo_yaml = generate_yolo_yaml(self.config, "yolov8n", {"yolov8n": self.model_info})
        if yolo_yaml is None:
            raise RuntimeError("generate_yolo_yaml returned None - no datasets configured for yolov8n.")

        total_epochs = int(self.args.epochs or self.model_info.get("epochs", 300))
        requested_batch = int(self.args.batch_size) if (self.args.batch_size and self.args.batch_size != "auto") else None
        res_ladder = self.opt_config.get("res_ladder", [320, 480, 640])
        start_res = getattr(self.args, "resolution", None)

        curriculum_stages = build_ladder_curriculum(
            res_ladder=res_ladder,
            total_epochs=total_epochs,
            vram_gb=vram_gb,
            requested_batch=requested_batch,
            opt_config=self.opt_config,
            start_resolution=start_res,
        )

        initial_checkpoint = self.project_root / "checkpoints" / "yolov8n" / "best.pt"
        if not initial_checkpoint.exists():
            initial_checkpoint = self.project_root / "checkpoints" / "yolov8n.pt"
        if not initial_checkpoint.exists():
            initial_checkpoint = self.project_root / "yolov8n.pt"

        current_weights_path = str(initial_checkpoint) if initial_checkpoint.exists() else "yolov8n.pt"

        print(
            f"[GOVERNOR] Commencing Governed YOLOv8n Training across {len(curriculum_stages)} ladder rungs | "
            f"VRAM Capacity: {vram_gb:.1f} GB | AMP Precision: {'FP16' if is_amp_safe else 'FP32 (Numerical Safe)'}",
            flush=True,
        )

        completed_prior_epochs = 0
        best_overall_metrics: Dict[str, float] = {"map50": 0.0, "map50_95": 0.0}
        sota_achieved = False

        for stage in curriculum_stages:
            print(
                f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(curriculum_stages)}] Launching {stage.resolution}px rung | "
                f"Batch Size: {stage.batch_size} (Sawtooth Guard) | Data Fraction: {stage.fraction*100:.0f}% | "
                f"Target Epochs: {stage.target_epochs} | Starting Weights: {Path(current_weights_path).name}",
                flush=True,
            )

            stage_dir = self.export_dir / f"stage{stage.stage_index}_{stage.resolution}px"
            stage_dir.mkdir(parents=True, exist_ok=True)

            model = YOLO(current_weights_path)

            stage_epochs_recorded = 0

            def create_epoch_callback(
                current_stage: YOLOLadderStage,
                prior_epochs: int,
            ) -> Callable[[Any], None]:
                def on_fit_epoch_end(trainer: Any) -> None:
                    nonlocal stage_epochs_recorded, best_overall_metrics, sota_achieved
                    try:
                        raw_epoch = int(getattr(trainer, "epoch", 0)) + 1
                        stage_epochs_recorded = raw_epoch
                        global_epoch = prior_epochs + raw_epoch

                        raw_metrics = getattr(trainer, "metrics", {}) or {}
                        map50 = float(raw_metrics.get("metrics/mAP50(B)", 0.0))
                        map50_95 = float(raw_metrics.get("metrics/mAP50-95(B)", 0.0))

                        box_loss = float(raw_metrics.get("val/box_loss", 0.0) or raw_metrics.get("train/box_loss", 0.0))
                        cls_loss = float(raw_metrics.get("val/cls_loss", 0.0) or raw_metrics.get("train/cls_loss", 0.0))
                        dfl_loss = float(raw_metrics.get("val/dfl_loss", 0.0) or raw_metrics.get("train/dfl_loss", 0.0))
                        val_loss = box_loss + cls_loss + dfl_loss
                        train_loss = float(getattr(trainer, "loss", 0.0) or val_loss)

                        lr_val = 0.01
                        if hasattr(trainer, "optimizer") and trainer.optimizer and trainer.optimizer.param_groups:
                            lr_val = float(trainer.optimizer.param_groups[0].get("lr", 0.01))

                        if map50 > best_overall_metrics.get("map50", 0.0):
                            best_overall_metrics["map50"] = map50
                        if map50_95 > best_overall_metrics.get("map50_95", 0.0):
                            best_overall_metrics["map50_95"] = map50_95

                        quality_score = (map50 * 50.0) + (map50_95 * 50.0)
                        gov_state = {
                            "input_size": current_stage.resolution,
                            "sample_fraction": current_stage.fraction,
                            "batch_size": current_stage.batch_size,
                            "accumulation_steps": 1,
                            "cooldown_remaining": 0,
                        }

                        curr_metrics = {
                            "map50": map50,
                            "map50_95": map50_95,
                            "box_loss": box_loss,
                            "cls_loss": cls_loss,
                            "dfl_loss": dfl_loss,
                        }

                        self.telemetry.write_epoch_row(
                            epoch=global_epoch - 1,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            lr=lr_val,
                            curr_metrics=curr_metrics,
                            quality_score=quality_score,
                            governor_state=gov_state,
                            stress=0.0,
                        )

                        # Synchronize metrics CSV to checkpoints and documentation models hub
                        csv_source = Path(self.telemetry.metrics_csv_path)
                        if csv_source.exists():
                            shutil.copy2(csv_source, self.models_hub_dir / "metrics.csv")

                        target_map50 = self.sota_targets.get("map50", 0.54)
                        target_map50_95 = self.sota_targets.get("map50_95", 0.39)
                        if current_stage.resolution >= 640 and current_stage.fraction >= 0.99:
                            if map50 >= target_map50 and map50_95 >= target_map50_95:
                                sota_achieved = True

                        if self.on_epoch_end is not None:
                            stream_payload = {
                                "epoch": float(global_epoch),
                                "map50": map50,
                                "map50_95": map50_95,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "resolution": float(current_stage.resolution),
                                "data_fraction": float(current_stage.fraction),
                                "batch_size": float(current_stage.batch_size),
                                "governor_active": 1.0,
                            }
                            self.on_epoch_end(global_epoch, stream_payload)

                    except Exception as cb_err:
                        logger.debug("YOLO governor epoch callback failed: %s", cb_err)

                return on_fit_epoch_end

            model.add_callback("on_fit_epoch_end", create_epoch_callback(stage, completed_prior_epochs))

            train_kwargs: Dict[str, Any] = {
                "data": yolo_yaml,
                "epochs": stage.target_epochs,
                "imgsz": stage.resolution,
                "device": device_arg,
                "batch": stage.batch_size,
                "fraction": stage.fraction,
                "project": str(stage_dir),
                "name": f"rung_{stage.resolution}",
                "exist_ok": True,
                "amp": is_amp_safe,
                "patience": stage.patience,
                "save": True,
                "plots": True,
                "verbose": True,
            }
            if getattr(self.args, "lr", None) is not None:
                train_kwargs["lr0"] = float(self.args.lr)

            model.train(**train_kwargs)

            # Locate stage checkpoint weights
            stage_weights_candidates = [
                stage_dir / f"rung_{stage.resolution}" / "weights" / "best.pt",
                stage_dir / f"rung_{stage.resolution}" / "weights" / "last.pt",
                stage_dir / "weights" / "best.pt",
            ]
            next_weights = None
            for cand in stage_weights_candidates:
                if cand.exists():
                    next_weights = str(cand)
                    break

            if next_weights:
                current_weights_path = next_weights
                print(
                    f"[GOVERNOR] [STAGE {stage.stage_index} COMPLETE] Advanced checkpoint: {Path(current_weights_path).name}",
                    flush=True,
                )
            else:
                logger.warning("Stage %d weights could not be located; retaining previous checkpoint.", stage.stage_index)

            completed_prior_epochs += stage_epochs_recorded

            if sota_achieved:
                print(
                    f"[GOVERNOR] [SOTA ATTAINED] Targets met on 640px full manifold! "
                    f"mAP50={best_overall_metrics['map50']:.4f}, mAP50-95={best_overall_metrics['map50_95']:.4f}. Concluding early.",
                    flush=True,
                )
                break

        # Final Canonical Checkpoint Consolidation
        final_best_source = Path(current_weights_path)
        canonical_export_weights = self.export_dir / "yolov8n" / "weights" / "best.pt"
        canonical_export_weights.parent.mkdir(parents=True, exist_ok=True)
        if final_best_source.exists():
            shutil.copy2(final_best_source, canonical_export_weights)
            shutil.copy2(final_best_source, self.checkpoints_dir / "best.pt")
            shutil.copy2(final_best_source, self.checkpoints_dir / "best.pth")
            shutil.copy2(final_best_source, self.models_hub_dir / "best.pt")

        print("[GOVERNOR] Training complete. Packaging final ONNX artifact at 640px SOTA resolution...", flush=True)
        try:
            export_model = YOLO(str(canonical_export_weights if canonical_export_weights.exists() else current_weights_path))
            export_model.export(format="onnx", imgsz=640)
            print("[GOVERNOR] ONNX export complete.", flush=True)
        except Exception as export_err:
            logger.warning("YOLO ONNX export encountered error: %s", export_err)

        total_time = round(time.time() - start_time, 2)
        print(f"[SUCCESS] Governed YOLOv8n multi-stage training finished in {total_time}s.", flush=True)

        return TrainingSummary(
            model_name="yolov8n",
            final_epoch=completed_prior_epochs,
            best_metrics=best_overall_metrics,
            total_time=total_time,
            status="completed",
        )


def run_governed_yolo_training(
    args: argparse.Namespace,
    config: Dict[str, Any],
    project_root: Path,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> TrainingSummary:
    """Entry point for governed YOLOv8n training dispatch."""
    governor = YOLOCurriculumGovernor(
        args=args,
        config=config,
        project_root=project_root,
        on_epoch_end=on_epoch_end,
    )
    return governor.run()
