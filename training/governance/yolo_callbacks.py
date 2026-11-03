"""Ultralytics callbacks installed by the YOLOv8n curriculum governor.

Cancellation hooks, the pretrain-routine and epoch-start hooks, and ``GovernedEpochHook``,
the fit-epoch-end hook that records telemetry, mirrors checkpoints, detects SOTA and, on
ladder rungs, decides when the rung advances (``RungAdvanceMonitor``). Rungs advance on a
quality plateau or on overfitting, never on an epoch count, and only ever forward.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.governance.yolo_ladder import YOLOLadderStage
from training.telemetry import EpochLosses, EpochRow
from training.utils.error_types import EPOCH_CALLBACK_ERRORS

if TYPE_CHECKING:
    from training.governance.yolo_governor import YOLOCurriculumGovernor

logger = logging.getLogger("lemtrain.governance.yolo")

# Failure modes of one governed epoch callback: metrics/CSV parsing, checkpoint and
# filesystem sync, and the caller-supplied on_epoch_end (formatting, log broadcast).
# Anything outside this set is a bug and propagates out of the Ultralytics fit loop.
_EPOCH_CALLBACK_ERRORS = (*EPOCH_CALLBACK_ERRORS, IndexError, ZeroDivisionError)

TRAINING_CANCELLED = "Training cancelled by user request."
VALIDATION_CANCELLED = "Validation cancelled by user request."


def stop_if_cancelled(
    cancel_check: Callable[[], bool] | None, target: Any, message: str = TRAINING_CANCELLED
) -> None:
    """Flag ``target`` to stop and raise ``InterruptedError`` when cancellation is requested."""
    if cancel_check is not None and cancel_check():
        if hasattr(target, "stop"):
            target.stop = True
        raise InterruptedError(message)


def make_cancel_hook(
    cancel_check: Callable[[], bool] | None, message: str = TRAINING_CANCELLED
) -> Callable[[Any], None]:
    """Build a trainer/validator hook that aborts the fit loop on cancellation."""

    def on_event(target: Any) -> None:
        stop_if_cancelled(cancel_check, target, message)

    return on_event


def restore_trainer_checkpoint_state(trainer: Any, ckpt: dict[str, Any]) -> None:
    """Restore optimizer, EMA and epoch state from ``ckpt`` into an Ultralytics trainer.

    Private-API adapter: Ultralytics ``BaseTrainer`` restores this state only through its
    private ``_load_checkpoint_state``. The public ``resume_training`` does nothing unless
    the trainer was started with ``resume=True``, and it also resets ``start_epoch`` and
    ``epochs``, which this governor sets itself for each curriculum stage. The method is
    looked up by name, so an Ultralytics release that drops it raises ``AttributeError``,
    which the caller already handles.
    """
    load_state = getattr(trainer, "_load_checkpoint_state", None)
    if load_state is None:
        raise AttributeError(
            f"{type(trainer).__name__} has no _load_checkpoint_state; "
            "this Ultralytics release cannot resume a curriculum stage in place"
        )
    load_state(ckpt)


def make_pretrain_routine_end(
    start_epoch: int, total_epochs: int, resume_ckpt: dict[str, Any] | None
) -> Callable[[Any], None]:
    """Build the hook that positions the trainer at the global epoch before training starts.

    With ``resume_ckpt`` the optimizer, EMA and epoch state are restored from it as well.
    """

    def on_pretrain_routine_end(trainer: Any) -> None:
        trainer.start_epoch = start_epoch
        trainer.epochs = total_epochs
        if resume_ckpt:
            try:
                restore_trainer_checkpoint_state(trainer, resume_ckpt)
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as load_err:
                logger.warning(
                    "Failed restoring checkpoint state in on_pretrain_routine_end: %s", load_err
                )
        if hasattr(trainer, "scheduler") and trainer.scheduler:
            trainer.scheduler.last_epoch = trainer.start_epoch - 1
        if hasattr(trainer, "validator") and trainer.validator:
            trainer.validator.get_desc = lambda: f"{'Validation':>15}"

    return on_pretrain_routine_end


def make_epoch_start_hook(
    default_total: int, describe: Callable[[int], str]
) -> Callable[[Any], None]:
    """Build the hook that keeps the epoch ceiling ahead and prints the epoch banner.

    ``describe`` maps the 1-based global epoch to the banner's rung description.
    """

    def on_train_epoch_start(trainer: Any) -> None:
        curr_global = int(getattr(trainer, "epoch", 0)) + 1
        # Epoch count is not a stage boundary: keep the global ceiling ahead of training
        if curr_global + 1 >= int(getattr(trainer, "epochs", default_total)):
            trainer.epochs = int(trainer.epochs) + 100
        print(
            f"\n[GOVERNOR] >>> Global Epoch {curr_global}/{int(trainer.epochs)} | "
            f"{describe(curr_global)}",
            flush=True,
        )

    return on_train_epoch_start


@dataclass(frozen=True)
class EpochMetrics:
    """Losses, mAP and learning rate Ultralytics reports at the end of one epoch."""

    map50: float
    map50_95: float
    box_loss: float
    cls_loss: float
    dfl_loss: float
    train_loss: float
    lr: float

    @property
    def val_loss(self) -> float:
        """Sum of the box, class and DFL losses."""
        return self.box_loss + self.cls_loss + self.dfl_loss

    @property
    def quality(self) -> float:
        """Combined quality used for plateau detection and best-model export."""
        return (self.map50_95 * 0.7) + (self.map50 * 0.3)

    @classmethod
    def from_trainer(cls, trainer: Any) -> EpochMetrics:
        """Read the epoch's metrics from an Ultralytics trainer (validation losses first)."""
        raw = getattr(trainer, "metrics", {}) or {}

        def loss(name: str) -> float:
            return float(raw.get(f"val/{name}", 0.0) or raw.get(f"train/{name}", 0.0))

        box_loss, cls_loss, dfl_loss = loss("box_loss"), loss("cls_loss"), loss("dfl_loss")
        lr_val = 0.01
        if hasattr(trainer, "optimizer") and trainer.optimizer and trainer.optimizer.param_groups:
            lr_val = float(trainer.optimizer.param_groups[0].get("lr", 0.01))
        return cls(
            map50=float(raw.get("metrics/mAP50(B)", 0.0)),
            map50_95=float(raw.get("metrics/mAP50-95(B)", 0.0)),
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss,
            train_loss=float(getattr(trainer, "loss", 0.0) or (box_loss + cls_loss + dfl_loss)),
            lr=lr_val,
        )


@dataclass
class RungAdvanceMonitor:
    """Plateau and overfitting detector that ends a ladder rung.

    Fraction and resolution progression is never tied to an epoch count. A rung advances
    only when combined mAP quality stops improving (plateau) or when train/val divergence
    indicates overfitting on the current data manifold.
    """

    min_epochs: int
    patience: int
    min_delta: float
    best: float = -1.0
    stall: int = 0
    rescued_overfitting: bool = False
    loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    map_history: list[float] = field(default_factory=list)

    def observe(
        self, trainer: Any, stage: YOLOLadderStage, metrics: EpochMetrics, rung_epochs: int
    ) -> None:
        """Record one epoch and set ``trainer.stop`` when the rung should advance."""
        self.loss_history.append(metrics.train_loss)
        self.val_loss_history.append(metrics.val_loss)
        self.map_history.append(metrics.map50)

        if metrics.quality > self.best + self.min_delta:
            self.best = metrics.quality
            self.stall = 0
        else:
            self.stall += 1

        if rung_epochs >= self.min_epochs and self.stall >= self.patience:
            print(
                f"\n[GOVERNOR] [PLATEAU DETECTED] No quality improvement for "
                f"{int(self.stall)} epochs at {stage.resolution}px @ "
                f"{stage.fraction * 100:.0f}% data (best quality "
                f"{self.best:.4f}). "
                f"Advancing to next ladder stage.",
                flush=True,
            )
            if hasattr(trainer, "stop"):
                trainer.stop = True

        if (
            rung_epochs >= self.min_epochs
            and len(self.loss_history) >= 4
            and not self.rescued_overfitting
        ):
            self._check_overfitting(trainer, stage)

    def _check_overfitting(self, trainer: Any, stage: YOLOLadderStage) -> None:
        losses, val_losses, maps = self.loss_history, self.val_loss_history, self.map_history
        train_trend = losses[-1] - losses[-3]
        val_trend = val_losses[-1] - val_losses[-3]
        val_consecutive_rise = (
            val_losses[-1] > val_losses[-2] > val_losses[-3] and losses[-1] <= losses[-2]
        )
        map_stagnant = (max(maps[-3:]) <= max(maps[:-3]) + 0.002) and (train_trend < -0.05)
        if not ((train_trend < -0.05 and val_trend > 0.05) or val_consecutive_rise or map_stagnant):
            return
        self.rescued_overfitting = True
        print(
            "\n[GOVERNOR] [OVERFITTING RESCUE] Overfitting attractor "
            "detected on "
            f"data manifold ({stage.fraction * 100:.0f}% @ "
            f"{stage.resolution}px rung). "
            f"Train Trend: {train_trend:+.4f} | Val Trend: "
            f"{val_trend:+.4f}. "
            "Halting stage early to expand dataset manifold and "
            "introduce sample variety!",
            flush=True,
        )
        if hasattr(trainer, "stop"):
            trainer.stop = True


@dataclass(frozen=True)
class EpochHookSpec:
    """Per-call-site behaviour of ``GovernedEpochHook``.

    ``summary_label`` heads the epoch summary line, ``default_total`` is the epoch ceiling
    shown when the trainer has none, ``top_res`` is the resolution a rung must reach before
    SOTA can end training (``None`` for SOTA extension cycles, which always qualify),
    ``monitor`` decides rung advancement (``None`` on extension cycles), ``cancel_first``
    checks cancellation before anything else, and ``failure_log`` names the hook in logs.
    """

    summary_label: str
    default_total: int
    failure_log: str
    top_res: int | None = None
    monitor: RungAdvanceMonitor | None = None
    cancel_first: bool = False


class GovernedEpochHook:
    """``on_fit_epoch_end`` hook: telemetry, checkpoint sync, SOTA and rung advancement."""

    def __init__(
        self,
        governor: YOLOCurriculumGovernor,
        stage: YOLOLadderStage,
        prior_epochs: int,
        target_dir: Path,
        spec: EpochHookSpec,
    ) -> None:
        self.governor = governor
        self.stage = stage
        self.prior_epochs = prior_epochs
        self.target_dir = target_dir
        self.spec = spec
        self.recorded: set[int] = set()
        self.epochs_recorded = 0

    def __call__(self, trainer: Any) -> None:
        """Process one finished epoch; expected failures are logged, never fatal."""
        gov = self.governor
        if self.spec.cancel_first:
            stop_if_cancelled(gov.cancel_check, trainer)
        try:
            curr_global = int(getattr(trainer, "epoch", 0)) + 1
            if curr_global in self.recorded:
                return
            self.recorded.add(curr_global)
            self.epochs_recorded = curr_global - self.prior_epochs
            metrics = EpochMetrics.from_trainer(trainer)
            self._record(trainer, curr_global, metrics)
            if self._sota_reached(metrics) and hasattr(trainer, "stop"):
                trainer.stop = True
            stop_if_cancelled(gov.cancel_check, trainer)
            if self.spec.monitor is not None:
                self.spec.monitor.observe(trainer, self.stage, metrics, self.epochs_recorded)
            if gov.on_epoch_end is not None:
                gov.on_epoch_end(curr_global, self._stream_payload(curr_global, metrics))
        except (InterruptedError, KeyboardInterrupt):
            raise
        except _EPOCH_CALLBACK_ERRORS:
            # Boundary: Ultralytics callback wrapping caller-supplied on_epoch_end;
            # an expected failure is logged with traceback and must not abort the
            # Ultralytics fit loop.
            logger.exception(self.spec.failure_log)

    def _record(self, trainer: Any, global_epoch: int, m: EpochMetrics) -> None:
        """Update best metrics, write the telemetry row, print the summary and sync."""
        gov, stage = self.governor, self.stage
        best = gov.best_overall_metrics
        if m.map50 > best.get("map50", 0.0):
            best["map50"] = m.map50
        if m.map50_95 > best.get("map50_95", 0.0):
            best["map50_95"] = m.map50_95
        gov.telemetry.write_epoch_row(
            EpochRow(
                epoch=global_epoch,
                losses=EpochLosses(train_loss=m.train_loss, val_loss=m.val_loss, lr=m.lr),
                metrics={
                    "map50": m.map50,
                    "map50_95": m.map50_95,
                    "box_loss": m.box_loss,
                    "cls_loss": m.cls_loss,
                    "dfl_loss": m.dfl_loss,
                },
                quality_score=(m.map50 * 50.0) + (m.map50_95 * 50.0),
                governor_state={
                    "input_size": stage.resolution,
                    "sample_fraction": stage.fraction,
                    "batch_size": stage.batch_size,
                    "accumulation_steps": 1,
                    "cooldown_remaining": 0,
                },
                stress=0.0,
            )
        )
        max_epochs_disp = int(getattr(trainer, "epochs", self.spec.default_total))
        print(
            f"\n[GOVERNOR] [{self.spec.summary_label} {global_epoch}/{max_epochs_disp} "
            "SUMMARY] "
            f"Train Loss: {m.train_loss:.4f} | Val Loss: {m.val_loss:.4f} "
            f"(Box: {m.box_loss:.4f}, Cls: {m.cls_loss:.4f}, Dfl: {m.dfl_loss:.4f}) | "
            f"mAP50: {m.map50:.4f} | mAP50-95: {m.map50_95:.4f}",
            flush=True,
        )
        # Synchronize checkpoints and metrics CSV to the models hub
        gov.synchronize_checkpoints(
            self.target_dir,
            stage.resolution,
            fraction=stage.fraction,
            stage_index=stage.stage_index,
            global_epoch=global_epoch,
        )

    def _sota_reached(self, m: EpochMetrics) -> bool:
        """Mark SOTA on the governor when targets are met on a qualifying rung."""
        gov, stage = self.governor, self.stage
        met = m.map50 >= gov.sota_targets.get("map50", 0.54) and m.map50_95 >= (
            gov.sota_targets.get("map50_95", 0.39)
        )
        if self.spec.top_res is not None:
            met = met and stage.resolution >= self.spec.top_res and stage.fraction >= 0.99
        if met:
            gov.sota_achieved = True
        return met

    def _stream_payload(self, global_epoch: int, m: EpochMetrics) -> dict[str, float]:
        stage = self.stage
        return {
            "epoch": float(global_epoch),
            "map50": m.map50,
            "map50_95": m.map50_95,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "resolution": float(stage.resolution),
            "data_fraction": float(stage.fraction),
            "batch_size": float(stage.batch_size),
            "governor_active": 1.0,
        }
