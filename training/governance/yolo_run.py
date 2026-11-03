"""Run orchestration for the YOLOv8n curriculum governor.

``YoloRunMixin.run`` plans the ladder, finds the resume point, trains each rung until it
plateaus or overfits, then runs open-ended SOTA extension cycles at the top rung and
packages the final artifacts. Resolution and data fraction only ever move forward: a
completed rung is skipped, never retrained at a lower setting.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.governance.yolo_callbacks import (
    TRAINING_CANCELLED,
    VALIDATION_CANCELLED,
    EpochHookSpec,
    GovernedEpochHook,
    RungAdvanceMonitor,
    make_cancel_hook,
    make_epoch_start_hook,
    make_pretrain_routine_end,
)
from training.governance.yolo_ladder import (
    BatchBudget,
    SawtoothMessages,
    YOLOLadderStage,
    build_ladder_curriculum,
    compute_safe_batch_size,
    is_amp_supported_for_device,
    sawtooth_batch,
)
from training.governance.yolo_resume import (
    MetricsHistory,
    ResumePoint,
    discover_checkpoint,
    load_resumable_state,
    replay_metrics_history,
    wipe_clean_run,
)
from training.training.summary import TrainingSummary

logger = logging.getLogger("lemtrain.governance.yolo")

_STAGE_SAWTOOTH = SawtoothMessages(
    pressure=(
        "\n[GOVERNOR] [SAWTOOTH SENTINEL] High VRAM pressure detected "
        "({pct:.1f}% >= {thresh:.0f}%). Dynamically reducing batch size: {old} -> {new}\n"
    ),
    headroom=(
        "\n[GOVERNOR] [SAWTOOTH DYNAMIC] Abundant VRAM headroom detected "
        "({pct:.1f}% < {thresh:.0f}% used). Dynamically scaling batch size: {old} -> {new}\n"
    ),
    probe_notice="Sawtooth dynamic memory probe notice: %s",
)
_EXTENSION_SAWTOOTH = SawtoothMessages(
    pressure=(
        "\n[GOVERNOR] [SAWTOOTH SENTINEL] VRAM pressure "
        "({pct:.1f}% >= {thresh:.0f}%). Downscaling extension batch size: {old} -> {new}\n"
    ),
    headroom=(
        "\n[GOVERNOR] [SAWTOOTH DYNAMIC] VRAM headroom "
        "({pct:.1f}% < {thresh:.0f}% used). Promoting extension batch size: {old} -> {new}\n"
    ),
    probe_notice="Sawtooth extension dynamic probe notice: %s",
)


@dataclass(frozen=True)
class _Hardware:
    """Training device string, primary-GPU VRAM and whether FP16 AMP is safe."""

    device_arg: str
    vram_gb: float
    is_amp_safe: bool


@dataclass
class _RunPlan:
    """Inputs fixed for one governed run: hardware, dataset YAML and ladder rungs."""

    hardware: _Hardware
    yolo_yaml: Any
    total_epochs: int
    requested_batch: int | None
    stages: list[YOLOLadderStage]
    initial_weights: str = "yolov8n.pt"

    @property
    def max_ladder_res(self) -> int:
        """Top rung resolution (640 when the curriculum is empty)."""
        return max(s.resolution for s in self.stages) if self.stages else 640


@dataclass
class _RunProgress:
    """Mutable run position: current weights, epochs done and the last rung trained."""

    weights: str
    completed_epochs: int = 0
    last_res: int = 0
    last_frac: float = 0.0


@dataclass(frozen=True)
class _EpochWindow:
    """Global start epoch and epoch ceiling of one rung, plus the state to resume from."""

    start_epoch: int
    total_epochs: int
    resume_ckpt: dict[str, Any] | None


def _first_nonempty(*candidates: Path) -> Path | None:
    """Return the first candidate file that exists and is not empty."""
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


class YoloRunMixin:
    """Curriculum run loop mixed into ``YOLOCurriculumGovernor``."""

    def _create_model(self, weights: str | Path) -> Any:
        raise NotImplementedError

    def _dataset_yaml(self) -> Any:
        raise NotImplementedError

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
        plan = self._plan_run()
        resume = self._load_resume_point(plan)
        progress = _RunProgress(weights=plan.initial_weights)
        for stage in plan.stages:
            if self._run_ladder_stage(plan, resume, progress, stage):
                return self._summary(progress, start_time, "cancelled")
            if self.sota_achieved:
                print(
                    "[GOVERNOR] [SOTA ATTAINED] Targets met on "
                    f"{plan.max_ladder_res}px full manifold! "
                    f"mAP50={self.best_overall_metrics['map50']:.4f}, "
                    f"mAP50-95={self.best_overall_metrics['map50_95']:.4f}. Concluding early.",
                    flush=True,
                )
                break
        if self._run_sota_extensions(plan, progress):
            return self._summary(progress, start_time, "cancelled")
        return self._finalize(progress, start_time)

    def _is_clean(self) -> bool:
        return bool(getattr(self.args, "clean", False))

    def _cancel_requested(self) -> bool:
        return self.cancel_check is not None and self.cancel_check()

    def _summary(self, progress: _RunProgress, start_time: float, status: str) -> TrainingSummary:
        return TrainingSummary(
            model_name="yolov8n",
            final_epoch=progress.completed_epochs,
            best_metrics=self.best_overall_metrics,
            total_time=round(time.time() - start_time, 2),
            status=status,
        )

    def _plan_run(self) -> _RunPlan:
        """Resolve hardware, the dataset YAML and the ladder curriculum for this run."""
        device_arg = self._resolve_devices()
        vram_gb = self._resolve_vram_gb()
        is_amp_safe = is_amp_supported_for_device(device_arg)

        yolo_yaml = self._dataset_yaml()
        if yolo_yaml is None:
            raise RuntimeError(
                "generate_yolo_yaml returned None - no datasets configured for yolov8n."
            )

        total_epochs = int(self.args.epochs or self.model_info.get("epochs", 300))
        requested_batch = (
            int(self.args.batch_size)
            if (self.args.batch_size and self.args.batch_size != "auto")
            else None
        )
        stages = build_ladder_curriculum(
            res_ladder=self.opt_config.get("res_ladder", [320, 480, 640]),
            total_epochs=total_epochs,
            budget=BatchBudget(vram_gb, requested_batch, self.opt_config),
            start_resolution=getattr(self.args, "resolution", None),
        )
        return _RunPlan(
            hardware=_Hardware(device_arg, vram_gb, is_amp_safe),
            yolo_yaml=yolo_yaml,
            total_epochs=total_epochs,
            requested_batch=requested_batch,
            stages=stages,
        )

    def _load_resume_point(self, plan: _RunPlan) -> ResumePoint:
        """Wipe or inspect prior state and print the run banner."""
        clean = self._is_clean()
        if clean:
            wipe_clean_run(self.models_hub_dir, self.models_hub_ckpt_dir)
        candidate_ckpt = (
            None if clean else discover_checkpoint(self.models_hub_dir, self.models_hub_ckpt_dir)
        )
        plan.initial_weights = str(candidate_ckpt) if candidate_ckpt else "yolov8n.pt"
        ckpt_dict = load_resumable_state(candidate_ckpt)

        history = MetricsHistory()
        metrics_csv_path = Path(self.telemetry.metrics_csv_path)
        if metrics_csv_path.exists() and not clean:
            history, self.best_export_quality = replay_metrics_history(
                metrics_csv_path,
                self.best_overall_metrics,
                self.best_export_quality,
                track_export_quality=(self.models_hub_dir / "yolov8n.pt").exists(),
            )

        hw = plan.hardware
        print(
            f"[GOVERNOR] Commencing Governed YOLOv8n Training across {len(plan.stages)} "
            "ladder rungs | "
            f"VRAM Capacity: {hw.vram_gb:.1f} GB | AMP Precision: "
            f"{'FP16' if hw.is_amp_safe else 'FP32 (Numerical Safe)'} | "
            f"Loaded Weights: {Path(plan.initial_weights).name} "
            f"(History: {history.csv_epochs} epochs)",
            flush=True,
        )
        return ResumePoint(ckpt_dict, history)

    def _run_ladder_stage(
        self,
        plan: _RunPlan,
        resume: ResumePoint,
        progress: _RunProgress,
        stage: YOLOLadderStage,
    ) -> bool:
        """Train one ladder rung (or skip it when already complete); True when cancelled."""
        progress.last_res, progress.last_frac = stage.resolution, stage.fraction
        frac_pct = round(stage.fraction * 100)
        # Stage output goes to the local runs directory, not the project root
        stage_dir = self.runs_dir / f"stage{stage.stage_index}_{stage.resolution}px_f{frac_pct}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"rung_{stage.resolution}" / "weights").mkdir(parents=True, exist_ok=True)

        if self._stage_already_complete(plan, resume, stage):
            prior_stage_epochs = resume.prior_epochs(stage, frac_pct, stage.target_epochs)
            print(
                f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(plan.stages)}] "
                f"{stage.resolution}px rung @ "
                f"{frac_pct}% data manifold already completed ({prior_stage_epochs} epochs). "
                f"Skipping to next ladder stage.",
                flush=True,
            )
            progress.completed_epochs += prior_stage_epochs
            return False

        stage.batch_size = sawtooth_batch(
            stage.batch_size,
            stage.resolution,
            BatchBudget(plan.hardware.vram_gb, plan.requested_batch, self.opt_config),
            _STAGE_SAWTOOTH,
        )
        stage_resume = resume.resumes(stage) and not self._is_clean()
        self._announce_stage(plan, resume, progress, stage, stage_resume)
        window = self._stage_window(plan, resume, progress, stage, stage_resume)
        model = self._create_model(progress.weights)
        epoch_hook = self._attach_stage_callbacks(model, plan, stage, progress, window)
        if self._train(
            model,
            self._train_kwargs(plan, stage, stage_dir, window.total_epochs),
            f"[GOVERNOR] Stage {stage.stage_index} interrupted by cancellation signal.",
            f"[GOVERNOR] Stage {stage.stage_index} cancellation signal detected. Halting.",
        ):
            return True

        self._advance_stage_weights(progress, stage, epoch_hook)
        progress.completed_epochs += (
            epoch_hook.epochs_recorded if epoch_hook.epochs_recorded > 0 else stage.target_epochs
        )
        resume.ckpt_dict = None  # Reset so subsequent ladder rungs start fresh with new resolution
        return False

    def _stage_already_complete(
        self, plan: _RunPlan, resume: ResumePoint, stage: YOLOLadderStage
    ) -> bool:
        """Whether a prior run finished ``stage``; the top full-data rung needs SOTA."""
        if (
            stage.resolution >= plan.max_ladder_res
            and stage.fraction >= 0.99
            and not self.sota_achieved
        ):
            return False
        return resume.stage_completed(stage) and not self._is_clean()

    def _announce_stage(
        self,
        plan: _RunPlan,
        resume: ResumePoint,
        progress: _RunProgress,
        stage: YOLOLadderStage,
        stage_resume: bool,
    ) -> None:
        """Print the rung banner; on resume, stage the checkpoint as the rung's last.pt."""
        frac_pct = round(stage.fraction * 100)
        header = f"\n[GOVERNOR] [STAGE {stage.stage_index}/{len(plan.stages)}] "
        if not stage_resume:
            print(
                f"{header}Launching {stage.resolution}px rung | "
                f"Batch Size: {stage.batch_size} (Sawtooth Guard) | Data Fraction: {frac_pct}% "
                "| "
                "Advance Rule: plateau/overfit (no epoch budget) | Starting Weights: "
                f"{Path(progress.weights).name}",
                flush=True,
            )
            return
        resume_target_epoch = max(resume.ckpt_epoch + 2, resume.history.csv_epochs + 1)
        print(
            f"{header}Resuming {stage.resolution}px rung from "
            f"global epoch {resume_target_epoch} | Batch Size: {stage.batch_size} | "
            f"Data Fraction: {frac_pct}% | Checkpoint: {Path(progress.weights).name}",
            flush=True,
        )
        stage_dir = self.runs_dir / f"stage{stage.stage_index}_{stage.resolution}px_f{frac_pct}"
        local_last = stage_dir / f"rung_{stage.resolution}" / "weights" / "last.pt"
        if not local_last.exists() and Path(progress.weights).exists():
            try:
                shutil.copy2(progress.weights, local_last)
            except OSError as exc:
                logger.warning(
                    "Could not stage %s as %s; resuming from source path: %s",
                    progress.weights,
                    local_last,
                    exc,
                )
        if local_last.exists():
            progress.weights = str(local_last)

    def _stage_window(
        self,
        plan: _RunPlan,
        resume: ResumePoint,
        progress: _RunProgress,
        stage: YOLOLadderStage,
        stage_resume: bool,
    ) -> _EpochWindow:
        """Global start epoch and ceiling of a rung; rebases ``progress`` on resume."""
        start_epoch = progress.completed_epochs
        if stage_resume and resume.ckpt_dict and resume.ckpt_epoch >= 0:
            start_epoch = resume.ckpt_epoch + 1
            progress.completed_epochs = max(
                0, start_epoch - resume.prior_epochs(stage, round(stage.fraction * 100), 0)
            )
        # Always anchor against metrics.csv historical progress if metrics.csv is further ahead
        # than checkpoint
        csv_epochs = resume.history.csv_epochs
        if not self._is_clean() and csv_epochs > 0:
            start_epoch = max(start_epoch, csv_epochs)
            progress.completed_epochs = max(progress.completed_epochs, csv_epochs)
        # Keep total_epochs ceiling dynamically ahead of the start epoch
        return _EpochWindow(
            start_epoch=start_epoch,
            total_epochs=max(plan.total_epochs, start_epoch + 100),
            resume_ckpt=resume.ckpt_dict if stage_resume else None,
        )

    def _attach_stage_callbacks(
        self,
        model: Any,
        plan: _RunPlan,
        stage: YOLOLadderStage,
        progress: _RunProgress,
        window: _EpochWindow,
    ) -> GovernedEpochHook:
        """Install a ladder rung's hooks on ``model`` and return its epoch-end hook."""
        frac_pct = round(stage.fraction * 100)
        prior_epochs = progress.completed_epochs
        monitor = RungAdvanceMonitor(
            min_epochs=int(self.opt_config.get("rung_min_epochs", 3)),
            patience=int(self.opt_config.get("rung_plateau_patience", 5)),
            min_delta=float(self.opt_config.get("rung_plateau_min_delta", 0.001)),
        )

        def describe(curr_global: int) -> str:
            return (
                f"Stage {stage.stage_index}/{len(plan.stages)} "
                f"({stage.resolution}px @ {frac_pct}% data | "
                f"Rung Epoch {curr_global - prior_epochs} | "
                f"Plateau Watch {int(monitor.stall)}/{monitor.patience})"
            )

        epoch_hook = GovernedEpochHook(
            self,
            stage,
            prior_epochs,
            self.runs_dir / f"stage{stage.stage_index}_{stage.resolution}px_f{frac_pct}",
            EpochHookSpec(
                summary_label="EPOCH",
                default_total=plan.total_epochs,
                failure_log="YOLO governor epoch callback failed",
                top_res=plan.max_ladder_res,
                monitor=monitor,
                cancel_first=True,
            ),
        )
        for event, message in (
            ("on_train_batch_start", TRAINING_CANCELLED),
            ("on_train_batch_end", TRAINING_CANCELLED),
            ("on_val_start", VALIDATION_CANCELLED),
            ("on_val_batch_start", VALIDATION_CANCELLED),
            ("on_val_batch_end", VALIDATION_CANCELLED),
        ):
            model.add_callback(event, make_cancel_hook(self.cancel_check, message))
        self._attach_epoch_hooks(model, window, describe, epoch_hook)
        return epoch_hook

    def _attach_epoch_hooks(
        self,
        model: Any,
        window: _EpochWindow,
        describe: Callable[[int], str],
        epoch_hook: GovernedEpochHook,
    ) -> None:
        """Install the pretrain, epoch-start, epoch-end and best-export hooks on ``model``."""
        model.add_callback(
            "on_pretrain_routine_end",
            make_pretrain_routine_end(window.start_epoch, window.total_epochs, window.resume_ckpt),
        )
        model.add_callback(
            "on_train_epoch_start", make_epoch_start_hook(window.total_epochs, describe)
        )
        model.add_callback("on_fit_epoch_end", epoch_hook)
        model.add_callback("on_model_save", self._export_if_new_best)

    def _train_kwargs(
        self, plan: _RunPlan, stage: YOLOLadderStage, project_dir: Path, epochs: int
    ) -> dict[str, Any]:
        """Ultralytics ``train`` arguments for one rung or extension cycle."""
        train_kwargs: dict[str, Any] = {
            "data": plan.yolo_yaml,
            "epochs": epochs,
            "imgsz": stage.resolution,
            "device": plan.hardware.device_arg,
            "batch": stage.batch_size,
            "fraction": stage.fraction,
            "project": str(project_dir),
            "name": f"rung_{stage.resolution}",
            "exist_ok": True,
            "amp": plan.hardware.is_amp_safe,
            "patience": stage.patience,
            "save": True,
            "plots": True,
            "verbose": True,
        }
        if getattr(self.args, "lr", None) is not None:
            train_kwargs["lr0"] = float(self.args.lr)
        return train_kwargs

    def _train(
        self, model: Any, train_kwargs: dict[str, Any], interrupted_msg: str, cancelled_msg: str
    ) -> bool:
        """Run ``model.train``; True (after printing why) when the run was cancelled."""
        try:
            model.train(**train_kwargs)
        except (InterruptedError, KeyboardInterrupt):
            print(interrupted_msg, flush=True)
            return True
        if self._cancel_requested():
            print(cancelled_msg, flush=True)
            return True
        return False

    def _advance_stage_weights(
        self, progress: _RunProgress, stage: YOLOLadderStage, epoch_hook: GovernedEpochHook
    ) -> None:
        """Carry the rung's best (else last) weights into the next rung and sync them."""
        frac_pct = round(stage.fraction * 100)
        stage_dir = epoch_hook.target_dir
        for rung in (f"rung_{stage.resolution}", f"rung_{stage.resolution}_f{frac_pct}", ""):
            weights_dir = stage_dir / rung / "weights" if rung else stage_dir / "weights"
            next_weights = _first_nonempty(weights_dir / "best.pt", weights_dir / "last.pt")
            if next_weights is not None:
                break
        if next_weights is None:
            logger.warning(
                "Stage %d weights could not be located; retaining previous checkpoint.",
                stage.stage_index,
            )
            return
        progress.weights = str(next_weights)
        self.synchronize_checkpoints(
            stage_dir,
            stage.resolution,
            fraction=stage.fraction,
            stage_index=stage.stage_index,
            global_epoch=progress.completed_epochs + epoch_hook.epochs_recorded,
        )
        print(
            f"[GOVERNOR] [STAGE {stage.stage_index} COMPLETE] Advanced checkpoint: "
            f"{Path(progress.weights).name}",
            flush=True,
        )

    def _run_sota_extensions(self, plan: _RunPlan, progress: _RunProgress) -> bool:
        """Open-Ended SOTA Convergence Protocol; True when cancelled mid-cycle.

        If the curriculum completed without reaching the SOTA targets on the top
        resolution at 100% data, training does not stop on a fixed epoch count: extension
        cycles at the top rung continue until SOTA is achieved or the operator cancels.
        """
        extension_cycle = 1
        while not self.sota_achieved:
            if self._cancel_requested():
                print(
                    "[GOVERNOR] Operator cancellation detected. Halting SOTA convergence loop.",
                    flush=True,
                )
                return False
            if self._run_extension_cycle(plan, progress, extension_cycle):
                return True
            extension_cycle += 1
        return False

    def _run_extension_cycle(self, plan: _RunPlan, progress: _RunProgress, cycle: int) -> bool:
        """Train one SOTA extension cycle at the top rung; True when cancelled."""
        top_res = plan.max_ladder_res
        extension_epochs = int(self.opt_config.get("extension_epochs", 20))
        print(
            f"\n[GOVERNOR] [SOTA CONVERGENCE LOOP - CYCLE {cycle}] "
            f"Top resolution ({top_res}px @ 100% data) continues until SOTA targets reached! "
            f"Current: mAP50={self.best_overall_metrics['map50']:.4f}/"
            f"{self.sota_targets.get('map50', 0.54)}, "
            f"mAP50-95={self.best_overall_metrics['map50_95']:.4f}/"
            f"{self.sota_targets.get('map50_95', 0.39)}. "
            f"Running extension cycle until plateau or SOTA (no epoch cap)...",
            flush=True,
        )
        ext_stage = YOLOLadderStage(
            stage_index=len(plan.stages) + cycle,
            resolution=top_res,
            batch_size=sawtooth_batch(
                compute_safe_batch_size(
                    imgsz=top_res,
                    vram_gb=plan.hardware.vram_gb,
                    requested_batch=plan.requested_batch,
                ),
                top_res,
                BatchBudget(plan.hardware.vram_gb, plan.requested_batch, self.opt_config),
                _EXTENSION_SAWTOOTH,
            ),
            fraction=1.0,
            target_epochs=extension_epochs,
            patience=max(10, int(self.opt_config.get("plateau_patience", 10))),
        )
        # Extension cycle output also goes to the local runs directory
        ext_dir = self.runs_dir / f"stage{ext_stage.stage_index}_{top_res}px_sota_ext{cycle}"
        ext_dir.mkdir(parents=True, exist_ok=True)
        ext_weights_dir = ext_dir / f"rung_{top_res}" / "weights"
        ext_weights_dir.mkdir(parents=True, exist_ok=True)

        window = _EpochWindow(
            progress.completed_epochs,
            max(plan.total_epochs, progress.completed_epochs + 100),
            None,
        )
        model = self._create_model(progress.weights)
        epoch_hook = GovernedEpochHook(
            self,
            ext_stage,
            window.start_epoch,
            ext_dir,
            EpochHookSpec(
                summary_label="SOTA EXT EPOCH",
                default_total=window.total_epochs,
                failure_log="YOLO governor extension callback failed",
            ),
        )
        model.add_callback("on_train_batch_end", make_cancel_hook(self.cancel_check))
        self._attach_epoch_hooks(
            model,
            window,
            lambda curr_global: (
                f"SOTA Extension Cycle {cycle} "
                f"({top_res}px @ 100% data | Ext Epoch {curr_global - window.start_epoch})"
            ),
            epoch_hook,
        )
        if self._train(
            model,
            self._train_kwargs(plan, ext_stage, ext_dir, window.total_epochs),
            f"[GOVERNOR] Extension cycle {cycle} interrupted by cancellation signal.",
            f"[GOVERNOR] Cancellation signal detected in extension cycle {cycle}. Halting.",
        ):
            return True

        ext_cand = _first_nonempty(ext_weights_dir / "best.pt", ext_weights_dir / "last.pt")
        if ext_cand is not None:
            progress.weights = str(ext_cand)
            self.synchronize_checkpoints(
                ext_dir,
                top_res,
                fraction=1.0,
                stage_index=ext_stage.stage_index,
                global_epoch=progress.completed_epochs + epoch_hook.epochs_recorded,
            )
        progress.last_res, progress.last_frac = top_res, 1.0
        progress.completed_epochs += (
            epoch_hook.epochs_recorded
            if epoch_hook.epochs_recorded > 0
            else ext_stage.target_epochs
        )
        return False

    def _finalize(self, progress: _RunProgress, start_time: float) -> TrainingSummary:
        """Consolidate the final checkpoint, export it and judge completion."""
        # Final Canonical Checkpoint Consolidation: all writes go to LemGendaryModels/yolov8n/
        final_best_source = Path(progress.weights)
        hub_ckpt_dir = self.models_hub_ckpt_dir
        hub_ckpt_dir.mkdir(parents=True, exist_ok=True)
        if final_best_source.exists():
            shutil.copy2(final_best_source, hub_ckpt_dir / "best.pt")
            shutil.copy2(final_best_source, hub_ckpt_dir / "best.pth")
            print(
                "[GOVERNOR] Training complete. Packaging final production artifacts "
                "(PyTorch & ONNX) to LemGendaryModels/yolov8n...",
                flush=True,
            )
            self.export_sota_models(final_best_source)

        total_time = round(time.time() - start_time, 2)
        print(
            f"[SUCCESS] Governed YOLOv8n multi-stage training finished in {total_time}s.",
            flush=True,
        )
        # Authoritative invariant check: minimum 20 epochs AND SOTA convergence attained, max
        # resolution ladder rung (640), and 100% data fraction
        res_ladder = self.opt_config.get("res_ladder", [320, 480, 640])
        max_ladder_res = max(res_ladder) if res_ladder else 640
        yolo_completed = (
            progress.completed_epochs >= 20
            and self.sota_achieved
            and progress.last_res >= max_ladder_res
            and progress.last_frac >= 0.999
        )
        return TrainingSummary(
            model_name="yolov8n",
            final_epoch=progress.completed_epochs,
            best_metrics=self.best_overall_metrics,
            total_time=total_time,
            status="completed" if yolo_completed else "partially_trained",
        )
