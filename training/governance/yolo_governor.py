"""Governed YOLOv8n Training Orchestrator and Curriculum Governor.

Coordinates multi-stage spatial resolution ladders (320px -> 480px -> 640px),
hardware-aware Sawtooth VRAM batch allocation, dataset fraction scaling,
numerical AMP stabilization, and metric telemetry synchronization while
preserving native Ultralytics inner-loop optimizations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
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


def _build_gradual_fractions(start: float) -> List[float]:
    """Build fractions from start to 1.0 where every increment is between 0.15 and 0.20."""
    if start >= 0.99:
        return [1.0]
    span = 1.0 - start
    steps = max(1, round(span / 0.175))
    step_size = span / steps
    if step_size > 0.20:
        steps += 1
    elif step_size < 0.15 and steps > 1:
        steps -= 1
    fractions = [round(start + i * (span / steps), 2) for i in range(steps)]
    fractions.append(1.0)
    return sorted(list(set(fractions)))


def get_fractions_for_resolution(
    resolution: int,
    is_lowest_rung: bool,
    opt_config: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Generate gradual fraction progression in 15%-20% increments up to 100%.

    Rules:
      - Lowest resolution rung (<= 320px) starts from initial fraction (~30%)
        and increases gradually by 15%-20% per stage: [0.30, 0.50, 0.70, 0.85, 1.00].
      - Subsequent higher resolution rungs (480px, 640px) start from 50% data
        and increase gradually by 15%-20% per stage: [0.50, 0.65, 0.80, 1.00].
    """
    opt_config = opt_config or {}
    if is_lowest_rung and resolution <= 320:
        initial = float(opt_config.get("initial_fraction", 0.30))
        if abs(initial - 0.30) < 0.05:
            return [0.30, 0.50, 0.70, 0.85, 1.00]
        return _build_gradual_fractions(initial)
    else:
        next_res_start = float(opt_config.get("next_res_start_fraction", 0.50))
        if abs(next_res_start - 0.50) < 0.05:
            return [0.50, 0.65, 0.80, 1.00]
        return _build_gradual_fractions(next_res_start)


def build_ladder_curriculum(
    res_ladder: List[int],
    total_epochs: int,
    vram_gb: float,
    requested_batch: Optional[int],
    opt_config: Dict[str, Any],
    start_resolution: Optional[int] = None,
) -> List[YOLOLadderStage]:
    """Construct deterministic multi-stage resolution ladder curriculum.

    Implements gradual intra-resolution fraction progression in 15%-20% increments:
      - Lowest resolution rung (<= 320px) starts from initial fraction (30%) and
        gradually progresses: 30% -> 50% -> 70% -> 85% -> 100%.
      - Subsequent higher resolution rungs (480px, 640px) start from 50% data
        and gradually progress: 50% -> 65% -> 80% -> 100%.
    This prevents overfitting and breaks performance plateaus when scaling across spatial rungs.
    """
    opt_config = opt_config or {}
    rungs = sorted(list(set(res_ladder))) if res_ladder else [320, 480, 640]

    if start_resolution is not None and start_resolution in rungs:
        rungs = [r for r in rungs if r >= start_resolution]
        if not rungs:
            rungs = [start_resolution]

    patience_base = int(opt_config.get("plateau_patience", 10))
    lowest_ladder_res = min(res_ladder) if res_ladder else 320

    stage_specs: List[Tuple[int, float]] = []
    for rung in rungs:
        is_lowest = (rung == lowest_ladder_res)
        fracs = get_fractions_for_resolution(rung, is_lowest, opt_config)
        for f in fracs:
            stage_specs.append((rung, f))

    lowest_res = min(s[0] for s in stage_specs)
    # Stage weights scale with spatial resolution and data manifold fraction
    weights = [(res / lowest_res) * frac for res, frac in stage_specs]
    total_weight = sum(weights)
    epoch_allocations = [max(1, round(total_epochs * (w / total_weight))) for w in weights]
    diff = total_epochs - sum(epoch_allocations)
    if diff != 0:
        epoch_allocations[-1] = max(1, epoch_allocations[-1] + diff)

    stages: List[YOLOLadderStage] = []
    for idx, ((res, frac), ep_budget) in enumerate(zip(stage_specs, epoch_allocations)):
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
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.args = args
        self.config = config
        self.project_root = project_root
        self.on_epoch_end = on_epoch_end
        self.cancel_check = cancel_check

        unified_models_rel = config.get("unified_models", "unified_models_v2.yaml")
        unified_models_path = project_root / unified_models_rel
        registry = {}
        if unified_models_path.exists():
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
        self.models_hub_dir = (project_root.parent / "LemGendaryModels" / "yolov8n").resolve()
        self.models_hub_ckpt_dir = self.models_hub_dir / "checkpoints"

        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_hub_dir.mkdir(parents=True, exist_ok=True)
        self.models_hub_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Primary metrics and telemetry stream directly from LemGendaryModels/yolov8n
        self.telemetry = TelemetryEngine(export_dir=str(self.models_hub_dir), task_type="yolo")
        self.telemetry.validate_and_initialize_csv()

        self.best_overall_metrics: Dict[str, float] = {"map50": 0.0, "map50_95": 0.0}
        self.sota_achieved = False
        self.best_sota_quality = 0.0

    def _export_sota_models(self, source_weights: Path, resolution: int = 640) -> None:
        """Export high-performance SOTA model as ONNX and PyTorch artifacts.

        Exports production-ready deployment assets directly to LemGendaryModels/yolov8n/:
          - yolov8n.pt (native PyTorch state)
          - yolov8n.onnx (optimized ONNX inference graph)
        """
        if not source_weights.exists() or source_weights.stat().st_size == 0:
            logger.warning("Cannot export SOTA model: source weights %s missing or empty.", source_weights)
            return

        try:
            # 1. Export PyTorch model to LemGendaryModels/yolov8n/yolov8n.pt
            target_pt = self.models_hub_dir / "yolov8n.pt"
            shutil.copy2(source_weights, target_pt)
            print(f"[GOVERNOR] [SOTA EXPORT] PyTorch SOTA model saved to: {target_pt}", flush=True)

            # 2. Export ONNX model to LemGendaryModels/yolov8n/yolov8n.onnx
            from ultralytics import YOLO
            export_model = YOLO(str(source_weights))
            exported_onnx_path = export_model.export(
                format="onnx",
                imgsz=resolution,
                dynamic=False,
                simplify=True,
            )
            if exported_onnx_path and Path(exported_onnx_path).exists():
                target_onnx = self.models_hub_dir / "yolov8n.onnx"
                shutil.copy2(exported_onnx_path, target_onnx)
                print(f"[GOVERNOR] [SOTA EXPORT] ONNX SOTA model exported to: {target_onnx}", flush=True)
        except Exception as export_err:
            logger.warning("Error during SOTA model export: %s", export_err)

    def _synchronize_checkpoints(
        self,
        stage_dir: Path,
        resolution: int,
        fraction: float = 1.0,
        stage_index: int = 1,
        global_epoch: Optional[int] = None,
    ) -> None:
        """Mirror stage checkpoints to canonical LemGendaryModels checkpoints and suite paths.

        Maintains real-time parity with standard PyTorch model checkpointing by
        ensuring best.pt, best.pth, last.pt, and progress.pth are synchronized strictly to:
          - LemGendaryModels/yolov8n/checkpoints/
          - lemgendary-training-suite/checkpoints/yolov8n/
        """
        frac_pct = round(fraction * 100)
        stage_weights_candidates = [
            stage_dir / f"rung_{resolution}_f{frac_pct}" / "weights",
            stage_dir / f"rung_{resolution}" / "weights",
            stage_dir / "weights",
        ]
        stage_weights_dir = stage_weights_candidates[0]
        for cand in stage_weights_candidates:
            if cand.exists():
                stage_weights_dir = cand
                break

        hub_ckpt_dir = self.models_hub_ckpt_dir
        hub_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Synchronize best weights strictly to checkpoints subfolders
        best_cand = stage_weights_dir / "best.pt"
        if best_cand.exists() and best_cand.stat().st_size > 0:
            try:
                shutil.copy2(best_cand, self.checkpoints_dir / "best.pt")
                shutil.copy2(best_cand, self.checkpoints_dir / "best.pth")
                shutil.copy2(best_cand, hub_ckpt_dir / "best.pt")
                shutil.copy2(best_cand, hub_ckpt_dir / "best.pth")
            except OSError as copy_err:
                logger.debug("Non-fatal checkpoint copy error for best weights: %s", copy_err)

        # Synchronize last / progress weights strictly to checkpoints subfolders
        last_cand = stage_weights_dir / "last.pt"
        if last_cand.exists() and last_cand.stat().st_size > 0:
            try:
                shutil.copy2(last_cand, self.checkpoints_dir / "last.pt")
                shutil.copy2(last_cand, self.checkpoints_dir / "progress.pth")
                shutil.copy2(last_cand, hub_ckpt_dir / "last.pt")
                shutil.copy2(last_cand, hub_ckpt_dir / "progress.pth")
            except OSError as copy_err:
                logger.debug("Non-fatal checkpoint copy error for last weights: %s", copy_err)

        # Synchronize metrics CSV to checkpoints and documentation models hub
        csv_source = Path(self.telemetry.metrics_csv_path)
        if csv_source.exists():
            try:
                shutil.copy2(csv_source, self.checkpoints_dir / "metrics.csv")
                shutil.copy2(csv_source, self.models_hub_dir / "metrics.csv")
            except OSError as csv_err:
                logger.debug("Non-fatal metrics CSV sync error: %s", csv_err)

        # Synchronize curriculum state strictly to checkpoints subfolders
        if global_epoch is not None:
            curriculum_state = {
                "model_key": "yolov8n",
                "global_epoch": global_epoch,
                "resolution": resolution,
                "fraction": round(fraction, 2),
                "stage_index": stage_index,
                "best_metrics": self.best_overall_metrics,
                "sota_achieved": self.sota_achieved,
            }
            for state_path in [
                hub_ckpt_dir / "curriculum_state.json",
                self.checkpoints_dir / "curriculum_state.json",
            ]:
                try:
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(curriculum_state, f, indent=2)
                except OSError as state_err:
                    logger.debug("Non-fatal curriculum state save error: %s", state_err)

        # Automated Cloud Sync Trigger if operating under remote environment
        if global_epoch is not None and getattr(self.args, "auto_sync", False) and getattr(self.args, "env", "local") == "kaggle":
            try:
                from training.cloud_sync import trigger_cloud_sync
                trigger_cloud_sync("yolov8n", global_epoch, self.config)
            except Exception as sync_err:
                logger.debug("Non-fatal cloud sync trigger error: %s", sync_err)

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

        # Clean run handling
        if getattr(self.args, "clean", False):
            print("[GOVERNOR] [CLEAN] Clean run requested. Wiping prior checkpoints, state, and metrics...", flush=True)
            for p in [
                self.models_hub_dir / "metrics.csv",
                self.models_hub_dir / "curriculum_state.json",
                self.models_hub_ckpt_dir / "curriculum_state.json",
                self.models_hub_ckpt_dir / "last.pt",
                self.models_hub_ckpt_dir / "progress.pth",
                self.models_hub_ckpt_dir / "best.pt",
                self.models_hub_ckpt_dir / "best.pth",
                self.checkpoints_dir / "metrics.csv",
                self.checkpoints_dir / "curriculum_state.json",
                self.checkpoints_dir / "last.pt",
                self.checkpoints_dir / "progress.pth",
            ]:
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

        # Discover candidate checkpoint from LemGendaryModels or suite
        candidate_ckpt: Optional[Path] = None
        if not getattr(self.args, "clean", False):
            for c in [
                self.models_hub_ckpt_dir / "last.pt",
                self.models_hub_ckpt_dir / "progress.pth",
                self.models_hub_ckpt_dir / "best.pt",
                self.models_hub_ckpt_dir / "best.pth",
                self.models_hub_dir / "yolov8n.pt",
                self.checkpoints_dir / "last.pt",
                self.checkpoints_dir / "best.pt",
                self.project_root / "checkpoints" / "yolov8n.pt",
                self.project_root / "yolov8n.pt",
            ]:
                if c.exists() and c.stat().st_size > 0:
                    candidate_ckpt = c
                    break

        current_weights_path = str(candidate_ckpt) if candidate_ckpt else "yolov8n.pt"

        # Inspect candidate checkpoint metadata
        ckpt_dict: Optional[Dict[str, Any]] = None
        if candidate_ckpt and candidate_ckpt.name in ["last.pt", "best.pt", "progress.pth", "best.pth", "yolov8n.pt"]:
            try:
                loaded = torch.load(str(candidate_ckpt), map_location="cpu")
                if isinstance(loaded, dict) and "epoch" in loaded and loaded.get("optimizer") is not None:
                    ckpt_dict = loaded
            except Exception as exc:
                logger.debug("Failed reading checkpoint dict from %s: %s", candidate_ckpt, exc)

        # Inspect metrics.csv history
        csv_epochs = 0
        last_csv_res = None
        last_csv_frac = None
        csv_stage_epoch_counts: Dict[Tuple[int, int], int] = {}
        metrics_csv_path = Path(self.telemetry.metrics_csv_path)
        if metrics_csv_path.exists() and not getattr(self.args, "clean", False):
            try:
                with open(metrics_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ep = row.get("Epoch")
                        if ep and ep.isdigit():
                            csv_epochs = max(csv_epochs, int(ep))
                            res_val = row.get("Res")
                            data_val = row.get("Data")
                            if res_val and res_val.isdigit():
                                r_int = int(res_val)
                                last_csv_res = r_int
                                f_pct = round(float(data_val) * 100) if data_val else 100
                                last_csv_frac = float(data_val) if data_val else 1.0
                                key = (r_int, f_pct)
                                csv_stage_epoch_counts[key] = csv_stage_epoch_counts.get(key, 0) + 1
                        m50 = float(row.get("mAP50", 0.0) or 0.0)
                        m95 = float(row.get("mAP50-95", 0.0) or 0.0)
                        if m50 > self.best_overall_metrics["map50"]:
                            self.best_overall_metrics["map50"] = m50
                        if m95 > self.best_overall_metrics["map50_95"]:
                            self.best_overall_metrics["map50_95"] = m95
            except Exception as e:
                logger.debug("Error reading metrics.csv: %s", e)

        print(
            f"[GOVERNOR] Commencing Governed YOLOv8n Training across {len(curriculum_stages)} ladder rungs | "
            f"VRAM Capacity: {vram_gb:.1f} GB | AMP Precision: {'FP16' if is_amp_safe else 'FP32 (Numerical Safe)'} | "
            f"Loaded Weights: {Path(current_weights_path).name} (History: {csv_epochs} epochs)",
            flush=True,
        )

        completed_prior_epochs = 0
        ckpt_epoch = int(ckpt_dict.get("epoch", -1)) if ckpt_dict else -1
        ckpt_imgsz = int(ckpt_dict.get("train_args", {}).get("imgsz", 0)) if ckpt_dict else 0
        ckpt_fraction = float(ckpt_dict.get("train_args", {}).get("fraction", 1.0)) if ckpt_dict else 1.0
        max_ladder_res = max(s.resolution for s in curriculum_stages) if curriculum_stages else 640

        for stage in curriculum_stages:
            frac_pct = round(stage.fraction * 100)
            stage_dir = self.export_dir / f"stage{stage.stage_index}_{stage.resolution}px_f{frac_pct}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_weights_dir = stage_dir / f"rung_{stage.resolution}" / "weights"
            stage_weights_dir.mkdir(parents=True, exist_ok=True)

            # Check if this stage was already fully completed in previous runs
            stage_already_complete = False
            if ckpt_dict:
                if ckpt_imgsz > stage.resolution:
                    stage_already_complete = True
                elif ckpt_imgsz == stage.resolution:
                    if ckpt_fraction > (stage.fraction + 0.05):
                        stage_already_complete = True
            elif last_csv_res is not None:
                if last_csv_res > stage.resolution:
                    stage_already_complete = True
                elif last_csv_res == stage.resolution and last_csv_frac is not None:
                    if last_csv_frac > (stage.fraction + 0.05):
                        stage_already_complete = True

            # The final top-res @ 100% data stage cannot be skipped as complete unless SOTA targets are achieved
            if stage.resolution >= max_ladder_res and stage.fraction >= 0.99 and not self.sota_achieved:
                stage_already_complete = False

            prior_stage_epochs = csv_stage_epoch_counts.get((stage.resolution, frac_pct), stage.target_epochs)

            if stage_already_complete and not getattr(self.args, "clean", False):
                print(
                    f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(curriculum_stages)}] {stage.resolution}px rung @ "
                    f"{frac_pct}% data manifold already completed ({prior_stage_epochs} epochs). Skipping to next ladder stage.",
                    flush=True,
                )
                completed_prior_epochs += prior_stage_epochs
                continue

            # Check if this stage should resume mid-run
            stage_resume = False
            if (
                ckpt_dict
                and ckpt_imgsz == stage.resolution
                and abs(ckpt_fraction - stage.fraction) <= 0.05
                and ckpt_epoch >= 0
                and not getattr(self.args, "clean", False)
            ):
                stage_resume = True
                print(
                    f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(curriculum_stages)}] Resuming {stage.resolution}px rung from epoch "
                    f"global epoch {ckpt_epoch + 2} | Batch Size: {stage.batch_size} | "
                    f"Data Fraction: {frac_pct}% | Checkpoint: {Path(current_weights_path).name}",
                    flush=True,
                )
                local_last = stage_weights_dir / "last.pt"
                if not local_last.exists() and Path(current_weights_path).exists():
                    try:
                        shutil.copy2(current_weights_path, local_last)
                    except OSError:
                        pass
                if local_last.exists():
                    current_weights_path = str(local_last)
            else:
                print(
                    f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(curriculum_stages)}] Launching {stage.resolution}px rung | "
                    f"Batch Size: {stage.batch_size} (Sawtooth Guard) | Data Fraction: {frac_pct}% | "
                    f"Advance Rule: plateau/overfit (no epoch budget) | Starting Weights: {Path(current_weights_path).name}",
                    flush=True,
                )

            stage_start_epoch = completed_prior_epochs
            if stage_resume and ckpt_dict and ckpt_epoch >= 0:
                stage_start_epoch = ckpt_epoch + 1
                completed_prior_epochs = max(
                    0, stage_start_epoch - csv_stage_epoch_counts.get((stage.resolution, frac_pct), 0)
                )

            model = YOLO(current_weights_path)
            stage_epochs_recorded = 0
            recorded_stage_epochs = set()
            rung_patience = int(self.opt_config.get("rung_plateau_patience", 5))
            rung_min_epochs = int(self.opt_config.get("rung_min_epochs", 3))
            rung_min_delta = float(self.opt_config.get("rung_plateau_min_delta", 0.001))
            plateau_state: Dict[str, float] = {"best": -1.0, "stall": 0}

            def on_pretrain_routine_end(trainer: Any) -> None:
                trainer.start_epoch = stage_start_epoch
                trainer.epochs = total_epochs
                if stage_resume and ckpt_dict:
                    try:
                        trainer._load_checkpoint_state(ckpt_dict)
                    except Exception as load_err:
                        logger.debug("Failed restoring checkpoint state in on_pretrain_routine_end: %s", load_err)
                if hasattr(trainer, "scheduler") and trainer.scheduler:
                    trainer.scheduler.last_epoch = trainer.start_epoch - 1
                if hasattr(trainer, "validator") and trainer.validator:
                    trainer.validator.get_desc = lambda: f"{'Validation':>15}"

            def on_train_epoch_start(trainer: Any) -> None:
                curr_global = int(getattr(trainer, "epoch", 0)) + 1
                # Epoch count is not a stage boundary: keep the global ceiling ahead of training
                if curr_global + 1 >= int(getattr(trainer, "epochs", total_epochs)):
                    trainer.epochs = int(trainer.epochs) + 100
                rung_ep = curr_global - completed_prior_epochs
                print(
                    f"\n[GOVERNOR] >>> Global Epoch {curr_global}/{int(trainer.epochs)} | "
                    f"Stage {stage.stage_index}/{len(curriculum_stages)} "
                    f"({stage.resolution}px @ {frac_pct}% data | Rung Epoch {rung_ep} | "
                    f"Plateau Watch {int(plateau_state['stall'])}/{rung_patience})",
                    flush=True,
                )

            def create_epoch_callback(
                current_stage: YOLOLadderStage,
                prior_epochs: int,
            ) -> Callable[[Any], None]:
                stage_loss_history: List[float] = []
                stage_val_loss_history: List[float] = []
                stage_map_history: List[float] = []
                rescued_overfitting: bool = False

                def on_fit_epoch_end(trainer: Any) -> None:
                    nonlocal stage_epochs_recorded, rescued_overfitting
                    try:
                        curr_global = int(getattr(trainer, "epoch", 0)) + 1
                        if curr_global in recorded_stage_epochs:
                            return
                        recorded_stage_epochs.add(curr_global)

                        global_epoch = curr_global
                        stage_epochs_recorded = curr_global - prior_epochs

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

                        if map50 > self.best_overall_metrics.get("map50", 0.0):
                            self.best_overall_metrics["map50"] = map50
                        if map50_95 > self.best_overall_metrics.get("map50_95", 0.0):
                            self.best_overall_metrics["map50_95"] = map50_95

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

                        print(
                            f"\n[GOVERNOR] [EPOCH {global_epoch}/{int(getattr(trainer, 'epochs', total_epochs))} SUMMARY] "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                            f"(Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, Dfl: {dfl_loss:.4f}) | "
                            f"mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}",
                            flush=True,
                        )

                        # Synchronize checkpoints and metrics CSV to checkpoints and documentation models hub
                        self._synchronize_checkpoints(
                            stage_dir,
                            current_stage.resolution,
                            fraction=current_stage.fraction,
                            stage_index=current_stage.stage_index,
                            global_epoch=global_epoch,
                        )

                        target_map50 = self.sota_targets.get("map50", 0.54)
                        target_map50_95 = self.sota_targets.get("map50_95", 0.39)
                        quality = (map50_95 * 0.7) + (map50 * 0.3)
                        if map50 >= target_map50 and map50_95 >= target_map50_95:
                            if current_stage.resolution >= max_ladder_res and current_stage.fraction >= 0.99:
                                self.sota_achieved = True
                                if hasattr(trainer, "stop"):
                                    trainer.stop = True
                            if quality > self.best_sota_quality:
                                self.best_sota_quality = quality
                                sota_cand = stage_weights_dir / "best.pt"
                                if not (sota_cand.exists() and sota_cand.stat().st_size > 0):
                                    trainer_best = getattr(trainer, "best", None)
                                    if trainer_best and Path(trainer_best).exists() and Path(trainer_best).stat().st_size > 0:
                                        sota_cand = Path(trainer_best)
                                if not (sota_cand.exists() and sota_cand.stat().st_size > 0):
                                    sota_cand = stage_weights_dir / "last.pt"
                                if sota_cand.exists() and sota_cand.stat().st_size > 0:
                                    print(
                                        f"\n[GOVERNOR] [SOTA ATTAINED] New SOTA performance achieved "
                                        f"(mAP50={map50:.4f}>={target_map50}, mAP50-95={map50_95:.4f}>={target_map50_95})! "
                                        f"Exporting SOTA ONNX and PyTorch artifacts to {self.models_hub_dir}...",
                                        flush=True,
                                    )
                                    self._export_sota_models(sota_cand, resolution=current_stage.resolution)

                        if self.cancel_check is not None and self.cancel_check():
                            if hasattr(trainer, "stop"):
                                trainer.stop = True
                            raise InterruptedError("Training cancelled by user request.")

                        # PLATEAU / OVERFITTING DRIVEN RUNG ADVANCEMENT
                        # Fraction and resolution progression is never tied to an epoch count. A rung
                        # advances only when combined mAP quality stops improving (plateau) or when
                        # train/val divergence indicates overfitting on the current data manifold.
                        stage_loss_history.append(train_loss)
                        stage_val_loss_history.append(val_loss)
                        stage_map_history.append(map50)

                        if quality > plateau_state["best"] + rung_min_delta:
                            plateau_state["best"] = quality
                            plateau_state["stall"] = 0
                        else:
                            plateau_state["stall"] += 1

                        if (
                            stage_epochs_recorded >= rung_min_epochs
                            and plateau_state["stall"] >= rung_patience
                        ):
                            print(
                                f"\n[GOVERNOR] [PLATEAU DETECTED] No quality improvement for "
                                f"{int(plateau_state['stall'])} epochs at {current_stage.resolution}px @ "
                                f"{current_stage.fraction*100:.0f}% data (best quality {plateau_state['best']:.4f}). "
                                f"Advancing to next ladder stage.",
                                flush=True,
                            )
                            if hasattr(trainer, "stop"):
                                trainer.stop = True

                        if stage_epochs_recorded >= rung_min_epochs and len(stage_loss_history) >= 4 and not rescued_overfitting:
                            train_trend = stage_loss_history[-1] - stage_loss_history[-3]
                            val_trend = stage_val_loss_history[-1] - stage_val_loss_history[-3]
                            val_consecutive_rise = (
                                stage_val_loss_history[-1] > stage_val_loss_history[-2] > stage_val_loss_history[-3]
                                and stage_loss_history[-1] <= stage_loss_history[-2]
                            )
                            map_recent_max = max(stage_map_history[-3:])
                            map_prior_max = max(stage_map_history[:-3])
                            map_stagnant = (map_recent_max <= map_prior_max + 0.002) and (train_trend < -0.05)

                            is_overfitting = (
                                (train_trend < -0.05 and val_trend > 0.05)
                                or val_consecutive_rise
                                or map_stagnant
                            )

                            if is_overfitting:
                                rescued_overfitting = True
                                print(
                                    f"\n[GOVERNOR] [OVERFITTING RESCUE] Overfitting attractor detected on data manifold "
                                    f"({current_stage.fraction*100:.0f}% @ {current_stage.resolution}px rung). "
                                    f"Train Trend: {train_trend:+.4f} | Val Trend: {val_trend:+.4f}. "
                                    f"Halting stage early to expand dataset manifold and introduce sample variety!",
                                    flush=True,
                                )
                                if hasattr(trainer, "stop"):
                                    trainer.stop = True

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

                    except (InterruptedError, KeyboardInterrupt):
                        raise
                    except Exception as cb_err:
                        logger.debug("YOLO governor epoch callback failed: %s", cb_err)

                return on_fit_epoch_end

            def on_train_batch_end(trainer: Any) -> None:
                if self.cancel_check is not None and self.cancel_check():
                    if hasattr(trainer, "stop"):
                        trainer.stop = True

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_fit_epoch_end", create_epoch_callback(stage, completed_prior_epochs))

            train_kwargs: Dict[str, Any] = {
                "data": yolo_yaml,
                "epochs": total_epochs,
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

            try:
                model.train(**train_kwargs)
            except (InterruptedError, KeyboardInterrupt):
                print(f"[GOVERNOR] Stage {stage.stage_index} interrupted by cancellation signal.", flush=True)
                return TrainingSummary(
                    model_name="yolov8n",
                    final_epoch=completed_prior_epochs,
                    best_metrics=self.best_overall_metrics,
                    total_time=round(time.time() - start_time, 2),
                    status="cancelled",
                )

            if self.cancel_check is not None and self.cancel_check():
                print(f"[GOVERNOR] Stage {stage.stage_index} cancellation signal detected. Halting.", flush=True)
                return TrainingSummary(
                    model_name="yolov8n",
                    final_epoch=completed_prior_epochs,
                    best_metrics=self.best_overall_metrics,
                    total_time=round(time.time() - start_time, 2),
                    status="cancelled",
                )

            # Locate stage checkpoint weights
            stage_weights_candidates = [
                stage_dir / f"rung_{stage.resolution}" / "weights" / "best.pt",
                stage_dir / f"rung_{stage.resolution}" / "weights" / "last.pt",
                stage_dir / f"rung_{stage.resolution}_f{frac_pct}" / "weights" / "best.pt",
                stage_dir / f"rung_{stage.resolution}_f{frac_pct}" / "weights" / "last.pt",
                stage_dir / "weights" / "best.pt",
                stage_dir / "weights" / "last.pt",
            ]
            next_weights = None
            for cand in stage_weights_candidates:
                if cand.exists() and cand.stat().st_size > 0:
                    next_weights = str(cand)
                    break

            if next_weights:
                current_weights_path = next_weights
                self._synchronize_checkpoints(
                    stage_dir,
                    stage.resolution,
                    fraction=stage.fraction,
                    stage_index=stage.stage_index,
                    global_epoch=completed_prior_epochs + stage_epochs_recorded,
                )
                print(
                    f"[GOVERNOR] [STAGE {stage.stage_index} COMPLETE] Advanced checkpoint: {Path(current_weights_path).name}",
                    flush=True,
                )
            else:
                logger.warning("Stage %d weights could not be located; retaining previous checkpoint.", stage.stage_index)

            completed_prior_epochs += stage_epochs_recorded
            ckpt_dict = None  # Reset so subsequent ladder rungs start fresh with new resolution

            if self.sota_achieved:
                print(
                    f"[GOVERNOR] [SOTA ATTAINED] Targets met on {max_ladder_res}px full manifold! "
                    f"mAP50={self.best_overall_metrics['map50']:.4f}, mAP50-95={self.best_overall_metrics['map50_95']:.4f}. Concluding early.",
                    flush=True,
                )
                break

        # Open-Ended SOTA Convergence Protocol:
        # If curriculum stages completed but SOTA targets on top resolution @ 100% data
        # have not yet been reached, do NOT stop on fixed epochs. Extend training until SOTA is achieved.
        top_res = max_ladder_res
        extension_cycle = 1
        target_map50 = self.sota_targets.get("map50", 0.54)
        target_map50_95 = self.sota_targets.get("map50_95", 0.39)

        while not self.sota_achieved:
            if self.cancel_check is not None and self.cancel_check():
                print("[GOVERNOR] Operator cancellation detected. Halting SOTA convergence loop.", flush=True)
                break

            extension_epochs = int(self.opt_config.get("extension_epochs", 20))
            print(
                f"\n[GOVERNOR] [SOTA CONVERGENCE LOOP - CYCLE {extension_cycle}] "
                f"Top resolution ({top_res}px @ 100% data) continues until SOTA targets reached! "
                f"Current: mAP50={self.best_overall_metrics['map50']:.4f}/{target_map50}, "
                f"mAP50-95={self.best_overall_metrics['map50_95']:.4f}/{target_map50_95}. "
                f"Running extension cycle until plateau or SOTA (no epoch cap)...",
                flush=True,
            )

            ext_stage = YOLOLadderStage(
                stage_index=len(curriculum_stages) + extension_cycle,
                resolution=top_res,
                batch_size=compute_safe_batch_size(
                    imgsz=top_res,
                    vram_gb=vram_gb,
                    requested_batch=requested_batch,
                ),
                fraction=1.0,
                target_epochs=extension_epochs,
                patience=max(10, int(self.opt_config.get("plateau_patience", 10))),
            )

            ext_dir = self.export_dir / f"stage{ext_stage.stage_index}_{top_res}px_sota_ext{extension_cycle}"
            ext_dir.mkdir(parents=True, exist_ok=True)
            ext_weights_dir = ext_dir / f"rung_{top_res}" / "weights"
            ext_weights_dir.mkdir(parents=True, exist_ok=True)

            ext_start_epoch = completed_prior_epochs
            ext_total_epochs = max(total_epochs, ext_start_epoch + 100)

            model = YOLO(current_weights_path)
            ext_stage_epochs_recorded = 0
            ext_recorded_epochs = set()

            def on_ext_pretrain_routine_end(trainer: Any) -> None:
                trainer.start_epoch = ext_start_epoch
                trainer.epochs = ext_total_epochs
                if hasattr(trainer, "scheduler") and trainer.scheduler:
                    trainer.scheduler.last_epoch = trainer.start_epoch - 1
                if hasattr(trainer, "validator") and trainer.validator:
                    trainer.validator.get_desc = lambda: f"{'Validation':>15}"

            def on_ext_train_epoch_start(trainer: Any) -> None:
                curr_global = int(getattr(trainer, "epoch", 0)) + 1
                if curr_global + 1 >= int(getattr(trainer, "epochs", ext_total_epochs)):
                    trainer.epochs = int(trainer.epochs) + 100
                ext_ep = curr_global - ext_start_epoch
                print(
                    f"\n[GOVERNOR] >>> Global Epoch {curr_global}/{int(trainer.epochs)} | "
                    f"SOTA Extension Cycle {extension_cycle} "
                    f"({top_res}px @ 100% data | Ext Epoch {ext_ep})",
                    flush=True,
                )

            def create_ext_callback(
                current_stage: YOLOLadderStage,
            ) -> Callable[[Any], None]:
                def on_ext_epoch_end(trainer: Any) -> None:
                    nonlocal ext_stage_epochs_recorded
                    try:
                        curr_global = int(getattr(trainer, "epoch", 0)) + 1
                        if curr_global in ext_recorded_epochs:
                            return
                        ext_recorded_epochs.add(curr_global)

                        global_epoch = curr_global
                        ext_stage_epochs_recorded = curr_global - ext_start_epoch

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

                        if map50 > self.best_overall_metrics.get("map50", 0.0):
                            self.best_overall_metrics["map50"] = map50
                        if map50_95 > self.best_overall_metrics.get("map50_95", 0.0):
                            self.best_overall_metrics["map50_95"] = map50_95

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

                        print(
                            f"\n[GOVERNOR] [SOTA EXT EPOCH {global_epoch}/{int(getattr(trainer, 'epochs', ext_total_epochs))} SUMMARY] "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                            f"(Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, Dfl: {dfl_loss:.4f}) | "
                            f"mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f}",
                            flush=True,
                        )

                        self._synchronize_checkpoints(
                            ext_dir,
                            current_stage.resolution,
                            fraction=current_stage.fraction,
                            stage_index=current_stage.stage_index,
                            global_epoch=global_epoch,
                        )

                        quality = (map50_95 * 0.7) + (map50 * 0.3)
                        if map50 >= target_map50 and map50_95 >= target_map50_95:
                            self.sota_achieved = True
                            if hasattr(trainer, "stop"):
                                trainer.stop = True
                            if quality > self.best_sota_quality:
                                self.best_sota_quality = quality
                                sota_cand = ext_weights_dir / "best.pt"
                                if not (sota_cand.exists() and sota_cand.stat().st_size > 0):
                                    trainer_best = getattr(trainer, "best", None)
                                    if trainer_best and Path(trainer_best).exists() and Path(trainer_best).stat().st_size > 0:
                                        sota_cand = Path(trainer_best)
                                if not (sota_cand.exists() and sota_cand.stat().st_size > 0):
                                    sota_cand = ext_weights_dir / "last.pt"
                                if sota_cand.exists() and sota_cand.stat().st_size > 0:
                                    print(
                                        f"\n[GOVERNOR] [SOTA ATTAINED] New SOTA performance achieved "
                                        f"(mAP50={map50:.4f}>={target_map50}, mAP50-95={map50_95:.4f}>={target_map50_95})! "
                                        f"Exporting SOTA ONNX and PyTorch artifacts to {self.models_hub_dir}...",
                                        flush=True,
                                    )
                                    self._export_sota_models(sota_cand, resolution=current_stage.resolution)

                        if self.cancel_check is not None and self.cancel_check():
                            if hasattr(trainer, "stop"):
                                trainer.stop = True
                            raise InterruptedError("Training cancelled by user request.")

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

                    except (InterruptedError, KeyboardInterrupt):
                        raise
                    except Exception as cb_err:
                        logger.debug("YOLO governor extension callback failed: %s", cb_err)

                return on_ext_epoch_end

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_pretrain_routine_end", on_ext_pretrain_routine_end)
            model.add_callback("on_train_epoch_start", on_ext_train_epoch_start)
            model.add_callback("on_fit_epoch_end", create_ext_callback(ext_stage))

            train_kwargs = {
                "data": yolo_yaml,
                "epochs": ext_total_epochs,
                "imgsz": ext_stage.resolution,
                "device": device_arg,
                "batch": ext_stage.batch_size,
                "fraction": 1.0,
                "project": str(ext_dir),
                "name": f"rung_{top_res}",
                "exist_ok": True,
                "amp": is_amp_safe,
                "patience": ext_stage.patience,
                "save": True,
                "plots": True,
                "verbose": True,
            }
            if getattr(self.args, "lr", None) is not None:
                train_kwargs["lr0"] = float(self.args.lr)

            try:
                model.train(**train_kwargs)
            except (InterruptedError, KeyboardInterrupt):
                print(f"[GOVERNOR] Extension cycle {extension_cycle} interrupted by cancellation signal.", flush=True)
                return TrainingSummary(
                    model_name="yolov8n",
                    final_epoch=completed_prior_epochs,
                    best_metrics=self.best_overall_metrics,
                    total_time=round(time.time() - start_time, 2),
                    status="cancelled",
                )

            if self.cancel_check is not None and self.cancel_check():
                print(f"[GOVERNOR] Cancellation signal detected in extension cycle {extension_cycle}. Halting.", flush=True)
                return TrainingSummary(
                    model_name="yolov8n",
                    final_epoch=completed_prior_epochs,
                    best_metrics=self.best_overall_metrics,
                    total_time=round(time.time() - start_time, 2),
                    status="cancelled",
                )

            ext_cand = ext_weights_dir / "best.pt"
            if not (ext_cand.exists() and ext_cand.stat().st_size > 0):
                ext_cand = ext_weights_dir / "last.pt"
            if ext_cand.exists() and ext_cand.stat().st_size > 0:
                current_weights_path = str(ext_cand)
                self._synchronize_checkpoints(
                    ext_dir,
                    top_res,
                    fraction=1.0,
                    stage_index=ext_stage.stage_index,
                    global_epoch=completed_prior_epochs + ext_stage_epochs_recorded,
                )

            completed_prior_epochs += ext_stage_epochs_recorded
            extension_cycle += 1

        # Final Canonical Checkpoint Consolidation
        final_best_source = Path(current_weights_path)
        canonical_export_weights = self.export_dir / "yolov8n" / "weights" / "best.pt"
        canonical_export_weights.parent.mkdir(parents=True, exist_ok=True)
        hub_ckpt_dir = self.models_hub_ckpt_dir
        hub_ckpt_dir.mkdir(parents=True, exist_ok=True)
        if final_best_source.exists():
            shutil.copy2(final_best_source, canonical_export_weights)
            shutil.copy2(final_best_source, self.checkpoints_dir / "best.pt")
            shutil.copy2(final_best_source, self.checkpoints_dir / "best.pth")
            shutil.copy2(final_best_source, hub_ckpt_dir / "best.pt")
            shutil.copy2(final_best_source, hub_ckpt_dir / "best.pth")

        export_source = canonical_export_weights if canonical_export_weights.exists() else final_best_source
        if export_source.exists():
            print("[GOVERNOR] Training complete. Packaging final production artifacts (PyTorch & ONNX) to LemGendaryModels/yolov8n...", flush=True)
            self._export_sota_models(export_source, resolution=640)

        total_time = round(time.time() - start_time, 2)
        print(f"[SUCCESS] Governed YOLOv8n multi-stage training finished in {total_time}s.", flush=True)

        return TrainingSummary(
            model_name="yolov8n",
            final_epoch=completed_prior_epochs,
            best_metrics=self.best_overall_metrics,
            total_time=total_time,
            status="completed",
        )


def run_governed_yolo_training(
    args: argparse.Namespace,
    config: Dict[str, Any],
    project_root: Path,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
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
