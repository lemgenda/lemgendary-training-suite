"""Main Training Engine Coordinator for LemGendary Training Suite.

Orchestrates multi-epoch training, validation passes, governance audits,
SOTA tracking, atomic checkpointing, rollback recovery, and cloud sync triggers.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch.utils.data.distributed import DistributedSampler

from data.forex_sampler import RowGroupAwareSampler
from training.checkpoint import CheckpointLifecycleManager, ExportSource, safe_atomic_save
from training.cloud.session_deadline import (
    DEADLINE_STATUS,
    SessionDeadlineReached,
    session_deadline,
    write_deadline_marker,
)
from training.cloud.sync_policy import KaggleSyncPolicy
from training.cloud_sync import trigger_cloud_sync
from training.data import rebuild_train_loader
from training.data.loaders import LoaderPlacement, TrainLoaderOptions, WorkerOverrides
from training.data.resumable_sampler import loader_batch_size, set_loader_epoch
from training.epoch_budget import (
    EpochAction,
    EpochBudgetConfig,
    EpochBudgetPolicy,
    StopReason,
)
from training.governance.curriculum import (
    DEFAULT_FOREX_LADDER,
    DEFAULT_SPATIAL_LADDER,
    is_forex_model,
)
from training.governance.epoch_audit import EpochSignals
from training.governance.logit_cap_sync import sync_logit_cap
from training.governance.measurement import unmeasured_reason
from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS
from training.ratchet import (
    FULL_FRACTION_THRESHOLD,
    executed_rung,
    top_resolution_reached,
)
from training.registry.task_type import resolve_task_type
from training.telemetry import EpochLosses, EpochRow, ForexRowTag, TelemetryEngine
from training.training.batch_budget import refit_batch_for_resolution
from training.training.context import TrainingContext
from training.training.epoch import train_one_epoch
from training.training.lr_scaling import scale_learning_rates
from training.training.step_progress import any_rank, is_main_process, progress_payload
from training.training.summary import TrainingSummary
from training.training.swa import attach_swa
from training.training.validation import validate_one_epoch
from training.utils.error_types import (
    CLOUD_SYNC_START_ERRORS,
    EPOCH_CALLBACK_ERRORS,
    STATE_RESTORE_ERRORS,
)

logger = logging.getLogger("lemtrain.engine")


def _rebuild_loader_on_directive(ctx: TrainingContext, f_chg: bool, r_chg: bool) -> None:
    """Apply the governor's fraction/resolution/batch to the dataset and rebuild the loader."""
    train_ds = getattr(ctx.data.train_loader, "dataset", None)
    if train_ds is not None and hasattr(train_ds, "update_strategy"):
        new_res = getattr(ctx.governance.governor, "current_res", None)
        new_frac = getattr(ctx.governance.governor, "current_fraction", None)
        stress_val = getattr(ctx.governance.governor, "current_stress", None)
        train_ds.update_strategy(
            size=new_res if r_chg else None,
            fraction=new_frac if f_chg else None,
            stress=stress_val,
        )
        if r_chg:
            refit_batch_for_resolution(ctx)
        fallback_batch = loader_batch_size(ctx.data.train_loader, 16)
        curr_batch = int(
            getattr(ctx.governance.governor.curriculum, "current_batch", fallback_batch)
        )
        ctx.data.accumulation_steps = max(
            1, int(getattr(ctx.governance.governor, "current_acc", ctx.data.accumulation_steps))
        )
        curr_sampler = getattr(ctx.data.train_loader, "sampler", None)
        if curr_sampler is not None and type(curr_sampler).__name__ == "RowGroupAwareSampler":
            curr_sampler = RowGroupAwareSampler(train_ds, shuffle=True, seed=42)
        elif curr_sampler is not None and type(curr_sampler).__name__ == "DistributedSampler":
            curr_sampler = DistributedSampler(train_ds, shuffle=True)
        elif curr_sampler is not None and type(curr_sampler).__name__ in (
            "RandomSampler",
            "SequentialSampler",
        ):
            curr_sampler = None
        ctx.data.train_loader = rebuild_train_loader(
            old_loader=ctx.data.train_loader,
            dataset=train_ds,
            batch_size=curr_batch,
            options=TrainLoaderOptions(
                sampler=curr_sampler,
                shuffle=(curr_sampler is None),
                workers=WorkerOverrides(
                    num_workers=ctx.run.args.num_workers
                    if hasattr(ctx.run.args, "num_workers")
                    else None
                ),
            ),
            placement=LoaderPlacement(
                device=ctx.hardware.device_info.device,
                env=getattr(ctx.run.args, "env", "local"),
                config=ctx.run.config,
            ),
        )
        print(
            f"[CURRICULUM] Updated train loader: {len(train_ds)} active samples "
            f"(fraction={float(new_frac or 1.0) * 100:.0f}%, res={new_res}, "
            f"batch_size={curr_batch}, total_batches={len(ctx.data.train_loader)})",
            flush=True,
        )
        logger.info(
            "Rebuilt train_loader on governor directive: res=%s, fraction=%s, batch_size=%d",
            str(new_res),
            str(new_frac),
            curr_batch,
        )


def _apply_lr_multiplier(ctx: TrainingContext) -> None:
    """Compose the governor's LR multipliers with the scheduler's decay."""
    governor = ctx.governance.governor
    lr_mult = float(getattr(governor, "lr_multiplier", 1.0))
    head_mult = float(getattr(governor, "head_lr_multiplier", lr_mult))
    scale_learning_rates(ctx.models.optimizer, ctx.models.scheduler, lr_mult, head_mult)
    logger.debug(
        "Applied governor learning rate multipliers: %.3fx (head %.3fx)", lr_mult, head_mult
    )


@dataclass
class _LoopState:
    """Cross-epoch bookkeeping of :func:`run_training`."""

    start_epoch: int
    started_at: float
    budget_policy: EpochBudgetPolicy
    best_metrics: dict[str, float] = field(default_factory=dict)
    epochs_no_improve: int = 0
    last_epoch_executed: int = 0
    stop_reason: StopReason | None = None


class _Hooks(NamedTuple):
    """Caller-supplied callbacks of :func:`run_training`."""

    on_epoch_end: Callable[[int, dict[str, float]], None] | None
    cancel_check: Callable[[], bool] | None

    def cancelled(self) -> bool:
        """Return True when the caller requested cancellation."""
        return self.cancel_check is not None and self.cancel_check()


@dataclass
class _EpochScore:
    """One epoch's merged metrics and the SOTA verdict derived from them."""

    train_metrics: dict[str, float]
    combined: dict[str, float]
    current_loss: float
    train_loss: float
    current_quality: float
    is_best: bool = False
    # Why the epoch cannot rank as a best (``None`` for a real measurement).
    unmeasured: str | None = None


class _GovernorSnapshot(NamedTuple):
    """Governor state that was active while the epoch trained (pre-audit)."""

    state: dict
    stress: float


class _AuditOutcome(NamedTuple):
    """The governor audit's verdict for the epoch budget policy."""

    early_stop: bool
    reports_improvement: bool


class _Pillars(NamedTuple):
    """The 4-Pillar Completion Invariant (minus the epoch-count pillar)."""

    ladder_passed: bool
    data_fraction_passed: bool
    sota_all_met: bool


def _is_forex(ctx: TrainingContext) -> bool:
    """Return True when the context trains the forex predictor family."""
    return is_forex_model(ctx.model_name, ctx.run.model_info)


def _resolve_start_epoch(ctx: TrainingContext) -> int:
    """Return the first epoch to run, honouring an (intra-epoch) resume."""
    if ctx.state.resume_state is not None and ctx.state.resume_state.epoch > 0:
        if getattr(ctx.state.resume_state, "is_intra_epoch", False):
            return ctx.state.resume_state.epoch
        return ctx.state.resume_state.epoch + 1
    return 1


def _prepare_run(ctx: TrainingContext, start_epoch: int) -> None:
    """Fix the epoch budget, create the checkpoint lifecycle manager and attach SWA."""
    # Ensure total_epochs is never lower than start_epoch
    if ctx.run.total_epochs < start_epoch:
        ctx.run.total_epochs = max(30, start_epoch + 10)

    if ctx.governance.lifecycle_manager is None:
        sota_targets = ctx.run.model_info.get("sota_targets", {})
        ctx.governance.lifecycle_manager = CheckpointLifecycleManager(
            model_name=ctx.model_name,
            project_root=ctx.run.paths.project_root,
            sota_targets=sota_targets,
        )

    ctx.governance.swa = attach_swa(ctx)
    ctx.state.session_deadline = session_deadline(
        ctx.run.config, str(getattr(ctx.run.args, "env", "local"))
    )


def _cancelled_summary(ctx: TrainingContext, loop: _LoopState, final_epoch: int) -> TrainingSummary:
    """Build the summary returned when the user cancels the run."""
    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=final_epoch,
        best_metrics=loop.best_metrics,
        total_time=time.time() - loop.started_at,
        status="cancelled",
    )


def _deadline_summary(
    ctx: TrainingContext, loop: _LoopState, stop: SessionDeadlineReached
) -> TrainingSummary:
    """End the run at the Kaggle session deadline: marker, synchronous Hub sync, summary.

    ``progress.pth`` (generator states included) was saved where the loop stopped, or
    ``latest.pth`` covers a stop at an epoch boundary. The final sync is not throttled and
    waits, so the payload is on the Hub before the session ends.
    """
    print(
        f"\n[DEADLINE] Kaggle session deadline reached for {ctx.model_name} at epoch "
        f"{stop.epoch} ({stop.phase} {stop.step}); progress saved, syncing and stopping.",
        flush=True,
    )
    if is_main_process():
        write_deadline_marker(ctx.run.paths.export_dir, stop.epoch, stop.phase, stop.step)
        if getattr(ctx.run.args, "auto_sync", False):
            try:
                trigger_cloud_sync(ctx.model_name, stop.epoch, ctx.run.config, wait=True)
            except CLOUD_SYNC_START_ERRORS as sync_err:
                logger.warning("Deadline cloud sync failed at epoch %d: %s", stop.epoch, sync_err)
    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=loop.last_epoch_executed,
        best_metrics=loop.best_metrics,
        total_time=time.time() - loop.started_at,
        status=DEADLINE_STATUS,
        stop_reason=DEADLINE_STATUS,
    )


def _deadline_before_epoch(ctx: TrainingContext) -> bool:
    """Whether the session deadline passed before an epoch starts (agreed by all ranks)."""
    deadline = ctx.state.session_deadline
    if deadline is None:
        return False
    return any_rank(deadline.expired(), ctx.hardware.device_info.device)


def _resumed_val_train_metrics(ctx: TrainingContext, epoch: int) -> dict[str, float]:
    """Return the train metrics of an epoch resumed directly into its validation pass."""
    ctx.state.guards.val_handled = True
    print(
        f"\n[RESUME] Resuming directly into validation pass for {ctx.model_name} at Epoch "
        f"{epoch} "
        "(progress checkpoint captured during validation phase; skipping redundant "
        "training pass)...",
        flush=True,
    )
    resume_tm = ctx.state.resume_state.train_metrics if ctx.state.resume_state is not None else {}
    train_metrics = dict(resume_tm) if resume_tm else {}
    if "train_loss" not in train_metrics:
        train_metrics["train_loss"] = float(train_metrics.get("loss", 0.0))
    if "epoch_time" not in train_metrics:
        train_metrics["epoch_time"] = 0.0
    return train_metrics


def _train_phase(
    ctx: TrainingContext, epoch: int, start_epoch: int, hooks: _Hooks
) -> dict[str, float]:
    """Run the pre-epoch governance step and the training pass (or skip it on a val resume)."""
    # 1. Pre-epoch governance step
    if hasattr(ctx.governance.governor, "thermal") and hasattr(
        ctx.governance.governor.thermal, "step_epoch"
    ):
        ctx.governance.governor.thermal.step_epoch()
    sync_logit_cap(ctx.models.raw_model or ctx.models.model, ctx.governance.governor)

    is_resuming_val = (
        ctx.state.resume_state is not None
        and ctx.state.resume_state.is_intra_epoch
        and ctx.state.resume_state.phase == "val"
        and epoch == start_epoch
        and not ctx.state.guards.val_handled
    )
    if is_resuming_val:
        return _resumed_val_train_metrics(ctx, epoch)

    if os.environ.get("RANK", "0") == "0":
        print(
            f"\n[EPOCH {epoch}/{ctx.run.total_epochs}] Initiating training pass for "
            f"{ctx.model_name}...",
            flush=True,
        )

    # Reseed the epoch permutation (ResumableBatchSampler, RowGroupAwareSampler, ...)
    set_loader_epoch(ctx.data.train_loader, epoch)

    # 2. Train one epoch
    return train_one_epoch(ctx, epoch, cancel_check=hooks.cancel_check)


def _score_epoch(
    ctx: TrainingContext, train_metrics: dict[str, float], val_metrics: dict[str, float]
) -> _EpochScore:
    """Merge the train/val metrics and derive the epoch's quality score."""
    # Combined metrics for governance and telemetry
    combined_metrics = {**train_metrics, **val_metrics}

    # 4. SOTA evaluation
    current_loss = float(combined_metrics.get("val_loss", 0.0))
    train_loss = float(combined_metrics.get("train_loss", 0.0))

    is_forex = _is_forex(ctx)
    task_type_str = resolve_task_type(ctx.run.model_info)
    sota_targets = ctx.run.model_info.get("sota_targets", {})

    if "quality_score" in combined_metrics:
        current_quality = float(combined_metrics["quality_score"])
    elif sota_targets:
        dummy_telemetry = TelemetryEngine(".", task_type="forex" if is_forex else "image")
        computed_qs, _ = dummy_telemetry.compute_quality_score(
            combined_metrics, sota_targets, task_type_str
        )
        current_quality = float(computed_qs)
    elif "val_time" in combined_metrics:
        # No SOTA targets: rank by the measured validation loss, mapped to (0, 1] so a
        # lower loss scores higher and every score stays positive.
        current_quality = 1.0 / (1.0 + max(0.0, current_loss))
    else:
        current_quality = math.nan
    unmeasured = unmeasured_reason(combined_metrics, sota_targets, current_quality, task_type_str)
    if unmeasured is not None:
        # Never a substitute score (the old ``100 - val_loss`` became a best): no quality.
        current_quality = math.nan
        logger.warning(
            "Epoch metrics are not a measurement (%s); it cannot rank as a best.", unmeasured
        )
    combined_metrics["quality_score"] = current_quality

    return _EpochScore(
        train_metrics=train_metrics,
        combined=combined_metrics,
        current_loss=current_loss,
        train_loss=train_loss,
        current_quality=current_quality,
        unmeasured=unmeasured,
    )


def _format_val_info(score: _EpochScore) -> str:
    """Render the validation part of the epoch summary line."""
    combined_metrics = score.combined
    val_info = f"ValLoss: {score.current_loss:.4f}"
    if "psnr" in combined_metrics:
        val_info += (
            f" | PSNR: {combined_metrics['psnr']:.2f}dB"
            f" | SSIM: {combined_metrics.get('ssim', 0.0):.4f}"
        )
        if "lpips" in combined_metrics:
            val_info += f" | LPIPS: {combined_metrics['lpips']:.4f}"
        if "fid" in combined_metrics:
            val_info += f" | FID: {combined_metrics['fid']:.2f}"
    if "plcc" in combined_metrics:
        val_info += (
            f" | PLCC: {combined_metrics['plcc']:.4f}"
            f" | SRCC: {combined_metrics.get('srcc', 0.0):.4f}"
        )
    if "dir_acc" in combined_metrics:
        win_rate = combined_metrics.get("win_rate", 0.0)
        val_info += f" | DirAcc: {combined_metrics['dir_acc']:.2f}% | WinRate: {win_rate:.2f}%"
    return val_info


def _print_epoch_summary(ctx: TrainingContext, epoch: int, score: _EpochScore) -> None:
    """Print the one-line epoch completion summary."""
    sota_star = " [NEW BEST]" if score.is_best else ""
    print(
        f"[EPOCH {epoch}/{ctx.run.total_epochs} COMPLETE] "
        f"TrainLoss: {score.train_loss:.4f} | "
        f"{_format_val_info(score)} | "
        f"Quality: {score.current_quality:.2f}{sota_star} | "
        f"EpochTime: {score.train_metrics.get('epoch_time', 0.0):.1f}s",
        flush=True,
    )


def _snapshot_governor(ctx: TrainingContext) -> _GovernorSnapshot:
    """Capture the governor state active during this epoch (telemetry + completion audit)."""
    governor = ctx.governance.governor
    pre_audit_gov_state = governor.get_state() if hasattr(governor, "get_state") else {}
    pre_audit_stress = float(getattr(governor, "current_stress", getattr(governor, "stress", 0.0)))
    return _GovernorSnapshot(state=pre_audit_gov_state, stress=pre_audit_stress)


def _record_epoch_telemetry(
    ctx: TrainingContext, epoch: int, score: _EpochScore, snapshot: _GovernorSnapshot
) -> None:
    """Record the epoch in the vault and export history/metrics CSV rows."""
    combined_metrics = score.combined
    lifecycle_manager = ctx.governance.lifecycle_manager
    ctx.governance.vault.record_epoch(
        epoch=epoch,
        metrics=combined_metrics,
        checkpoint_path=str(lifecycle_manager.best_path)
        if (score.is_best and lifecycle_manager)
        else None,
    )
    if ctx.run.paths.history_csv_path is not None:
        ctx.governance.vault.export_csv(ctx.run.paths.history_csv_path)

    # Standard LemGendary Telemetry Engine export (metrics.csv)
    if lifecycle_manager is None:
        return
    task_type = "forex" if _is_forex(ctx) else "image"
    telemetry = TelemetryEngine(
        export_dir=str(lifecycle_manager.model_hub_dir), task_type=task_type
    )
    quality_score = float(combined_metrics.get("quality_score", score.current_quality))
    fold = int(getattr(ctx.run.args, "fold", 1) or 1)
    phase = int(getattr(ctx.run.args, "phase", 1) or 1)
    pairs_cfg = ctx.run.model_info.get("kwargs", {}).get("pairs", [])
    num_pairs = len(pairs_cfg) if pairs_cfg else 16
    telemetry.write_epoch_row(
        EpochRow(
            epoch=epoch,
            losses=EpochLosses(
                train_loss=score.train_loss,
                val_loss=score.current_loss,
                lr=float(combined_metrics.get("lr", 0.0)),
            ),
            metrics=combined_metrics,
            quality_score=quality_score,
            governor_state=snapshot.state,
            stress=snapshot.stress,
            forex=ForexRowTag(num_pairs=num_pairs, phase=phase, fold=fold),
        )
    )


def _audit_governor(
    ctx: TrainingContext, epoch: int, score: _EpochScore, epochs_no_improve: int
) -> _AuditOutcome:
    """Run the governor audit and apply its loader/LR directives."""
    governor_early_stop = False
    governor_reports_improvement = False
    if not hasattr(ctx.governance.governor, "audit_epoch"):
        return _AuditOutcome(governor_early_stop, governor_reports_improvement)
    try:
        f_chg, r_chg, lr_chg, _t_chg, _c_chg, _b_chg, stop_trig, gov_msg = (
            ctx.governance.governor.audit_epoch(
                EpochSignals(
                    current_quality=score.current_quality,
                    current_loss=score.current_loss,
                    train_loss=score.train_loss,
                    metrics_dict=score.combined,
                ),
                epochs_no_improve=epochs_no_improve,
            )
        )
        if gov_msg:
            print(f"{gov_msg}", flush=True)
        governor_early_stop = stop_trig
        governor_reports_improvement = not governor_early_stop

        # Dynamic manifold scaling: update dataset and rebuild train_loader if fraction,
        # resolution, or batch changed
        if (f_chg or r_chg or _b_chg) and ctx.data.train_loader is not None:
            _rebuild_loader_on_directive(ctx, f_chg, r_chg)

        # Learning rate propulsion or cooling adjustments
        if lr_chg and ctx.models.optimizer is not None:
            _apply_lr_multiplier(ctx)

    except STATE_RESTORE_ERRORS as gov_err:
        logger.warning("Governor audit or loader rebuild failed at epoch %d: %s", epoch, gov_err)
    return _AuditOutcome(governor_early_stop, governor_reports_improvement)


_DEFAULT_DUMMY_SHAPE: tuple[int, ...] = (1, 3, 256, 256)


def _shape_from_size_list(raw_size: list) -> tuple[int, ...]:
    """Map a registry ``input_size`` list to a batch-of-one dummy-input shape."""
    if len(raw_size) == 3:
        return (1, int(raw_size[0]), int(raw_size[1]), int(raw_size[2]))
    if len(raw_size) == 2:
        return (1, 3, int(raw_size[0]), int(raw_size[1]))
    if len(raw_size) == 1:
        return (1, int(raw_size[0]))
    return _DEFAULT_DUMMY_SHAPE


def _onnx_dummy_shape(ctx: TrainingContext) -> tuple[int, ...]:
    """Return the dummy-input shape used for the ONNX export of a new best model."""
    raw_size = ctx.run.model_info.get("input_size")
    if isinstance(raw_size, list):
        return _shape_from_size_list(raw_size)
    if isinstance(raw_size, int):
        return (1, 3, raw_size, raw_size)
    dataset_tensors = (
        getattr(getattr(ctx.data.train_loader, "dataset", None), "tensors", ())
        if ctx.data.train_loader is not None
        else ()
    )
    if len(dataset_tensors) > 0:
        sample_tensor = dataset_tensors[0]
        return (1, *[int(d) for d in sample_tensor.shape[1:]])
    return _DEFAULT_DUMMY_SHAPE


def _save_checkpoints(
    ctx: TrainingContext, epoch: int, score: _EpochScore, best_metrics: dict[str, float]
) -> None:
    """Persist the latest/best checkpoints and vault milestones (post-audit governor state)."""
    raw_model = ctx.models.raw_model if ctx.models.raw_model is not None else ctx.models.model
    checkpoint_payload = progress_payload(
        ctx,
        epoch,
        raw_model,
        {
            "best_quality_score": float(ctx.governance.sota_tracker.best_quality),
            "best_metrics": best_metrics,
            "current_metrics": score.combined,
        },
    )

    lifecycle_manager = ctx.governance.lifecycle_manager
    if lifecycle_manager is None:
        if score.is_best and ctx.run.paths.best_checkpoint_path is not None:
            safe_atomic_save(checkpoint_payload, ctx.run.paths.best_checkpoint_path)
        if ctx.run.paths.history_csv_path is not None:
            ctx.governance.vault.export_csv(ctx.run.paths.history_csv_path)
        return

    lifecycle_manager.save_latest(epoch=epoch, payload=checkpoint_payload)
    if score.is_best:
        # Prepare dummy input for ONNX export
        dummy_input = torch.randn(*_onnx_dummy_shape(ctx), device=ctx.hardware.device_info.device)
        lifecycle_manager.save_best(
            epoch=epoch,
            quality_score=score.current_quality,
            payload=checkpoint_payload,
            export=ExportSource(model=raw_model, dummy_input=dummy_input),
        )

    # Check and save vault milestones (an unmeasured epoch has none to offer)
    if score.unmeasured is not None:
        return
    lifecycle_manager.check_and_save_vault(
        epoch=epoch,
        metrics=score.combined,
        payload=checkpoint_payload,
    )


def maybe_trigger_cloud_sync(ctx: TrainingContext, epoch: int, is_best: bool) -> None:
    """Trigger the Kaggle auto-sync when it is configured and the throttle allows it.

    The throttle (``fleet.kaggle_sync_every_epochs`` / ``fleet.kaggle_sync_on_best``) is
    :class:`training.cloud.sync_policy.KaggleSyncPolicy`.
    """
    args = ctx.run.args
    if not (getattr(args, "auto_sync", False) and getattr(args, "env", "local") == "kaggle"):
        return
    if not KaggleSyncPolicy.from_config(ctx.run.config).should_sync(epoch, is_best):
        logger.debug("Kaggle auto-sync throttled at epoch %d", epoch)
        return
    try:
        trigger_cloud_sync(ctx.model_name, epoch, ctx.run.config)
    except CLOUD_SYNC_START_ERRORS as sync_err:
        logger.warning("Automated cloud sync trigger failed at epoch %d: %s", epoch, sync_err)


def _notify_epoch_end(epoch: int, score: _EpochScore, hooks: _Hooks) -> bool:
    """Invoke the external epoch callback; return True when it signalled cancellation."""
    if hooks.on_epoch_end is None:
        return False
    try:
        hooks.on_epoch_end(epoch, score.combined)
    except (InterruptedError, KeyboardInterrupt):
        print(f"[CANCEL] Epoch {epoch} interrupted by cancellation callback.", flush=True)
        return True
    except EPOCH_CALLBACK_ERRORS:
        # Boundary: on_epoch_end is caller-supplied code; an expected failure is logged
        # with traceback and must not abort training.
        logger.exception("Telemetry epoch callback failed at epoch %d", epoch)
    return False


def _ladder_passed(ctx: TrainingContext, executed_res: int) -> bool:
    """Return True when the executed resolution reached the top ladder rung (Fold 6 for forex)."""
    opt_cfg = ctx.run.model_info.get("optimization", {})
    if _is_forex(ctx):
        res_ladder = (
            opt_cfg.get("res_ladder")
            or getattr(ctx.governance.governor, "res_ladder", None)
            or list(DEFAULT_FOREX_LADDER)
        )
        target_res = max(res_ladder) if res_ladder else DEFAULT_FOREX_LADDER[-1]
        return getattr(ctx.run.args, "fold", 1) >= 6 or executed_res >= target_res
    res_ladder = (
        opt_cfg.get("res_ladder")
        or getattr(ctx.governance.governor, "res_ladder", None)
        or list(DEFAULT_SPATIAL_LADDER)
    )
    return top_resolution_reached(executed_res, res_ladder)


def _sota_targets_met(ctx: TrainingContext, combined_metrics: dict[str, float]) -> bool:
    """Return True when every defined SOTA target is met by the epoch's metrics."""
    sota_targets = ctx.run.model_info.get("sota_targets", {})
    if not (isinstance(sota_targets, dict) and sota_targets):
        return True
    for s_key, s_target in sota_targets.items():
        if s_key not in combined_metrics:
            return False
        achieved = float(combined_metrics[s_key])
        target_val = float(s_target)
        is_higher = DEFAULT_METRIC_DIRECTIONS.get(s_key, True)
        if is_higher and achieved < target_val or not is_higher and achieved > target_val:
            return False
    return True


def _completion_pillars(
    ctx: TrainingContext, pre_audit_gov_state: dict, combined_metrics: dict[str, float]
) -> _Pillars:
    """Evaluate the ladder, data-fraction and SOTA pillars of the completion invariant.

    Reads the rung this epoch actually trained at (pre-audit state, the keys
    get_state() writes). Never falls back to the post-audit governor: a spatial
    jump during the audit must not count as training there.
    """
    executed_res, executed_frac = executed_rung(pre_audit_gov_state)
    return _Pillars(
        ladder_passed=_ladder_passed(ctx, executed_res),
        data_fraction_passed=executed_frac >= FULL_FRACTION_THRESHOLD,
        sota_all_met=_sota_targets_met(ctx, combined_metrics),
    )


def _apply_budget_decision(
    ctx: TrainingContext, loop: _LoopState, epoch: int, pillars: _Pillars, audit: _AuditOutcome
) -> None:
    """Ask the epoch budget policy whether to continue, extend or stop the run."""
    # Evaluate 4-Pillar Completion Invariant:
    # 1. Minimum 20 epochs
    # 2. Top resolution ladder rung (or Fold 6 for forex)
    # 3. 100% data fraction (>= 0.999) during the completed epoch
    # 4. All defined SOTA targets met
    min_epochs_met = loop.last_epoch_executed >= 20
    is_fully_completed = (
        min_epochs_met
        and pillars.ladder_passed
        and pillars.data_fraction_passed
        and pillars.sota_all_met
    )
    decision = loop.budget_policy.decide(
        epoch,
        converged=is_fully_completed,
        early_stop=audit.early_stop,
        governor_reports_improvement=audit.reports_improvement,
    )
    ctx.run.total_epochs = decision.total_epochs
    pillar_info = (
        f"(Epochs>=20: {min_epochs_met}, Ladder: {pillars.ladder_passed}, "
        f"Data: {pillars.data_fraction_passed}, SOTA: {pillars.sota_all_met})."
    )
    if decision.action is EpochAction.STOP:
        loop.stop_reason = decision.stop_reason
        reason_label = loop.stop_reason.value if loop.stop_reason is not None else "stop"
        print(
            f"\n[STOP] [{reason_label}] {ctx.model_name}: {decision.message} {pillar_info}",
            flush=True,
        )
        logger.info("Training loop stopped for %s: %s", ctx.model_name, decision.message)
    elif decision.action is EpochAction.EXTEND:
        print(
            f"\n[GOVERNOR] [BUDGET EXTENSION] {ctx.model_name}: {decision.message} {pillar_info}",
            flush=True,
        )
        logger.info("Epoch budget extended for %s: %s", ctx.model_name, decision.message)


def _run_epoch(
    ctx: TrainingContext, loop: _LoopState, epoch: int, hooks: _Hooks
) -> TrainingSummary | None:
    """Run one full epoch; return a summary only when the user cancelled mid-epoch."""
    train_metrics = _train_phase(ctx, epoch, loop.start_epoch, hooks)

    # Cache last train metrics on context for validation-phase progress saves
    ctx.state.last_train_metrics = train_metrics

    if hooks.cancelled():
        print(
            f"[CANCEL] Training aborted after epoch {epoch} training pass by user cancellation.",
            flush=True,
        )
        return _cancelled_summary(ctx, loop, epoch)

    # 3. Validate one epoch
    val_metrics = validate_one_epoch(ctx, epoch, cancel_check=hooks.cancel_check)
    score = _score_epoch(ctx, train_metrics, val_metrics)
    score.is_best = score.unmeasured is None and ctx.governance.sota_tracker.record_epoch(
        current_quality=score.current_quality,
        current_loss=score.current_loss,
        train_loss=score.train_loss,
    )
    if score.is_best:
        loop.best_metrics = dict(score.combined)
    _print_epoch_summary(ctx, epoch, score)

    # 5. Capture active governor state used during this epoch for telemetry and completion audit
    snapshot = _snapshot_governor(ctx)
    # 6. Vault recording and canonical telemetry export (records active epoch parameters)
    _record_epoch_telemetry(ctx, epoch, score, snapshot)
    loop.epochs_no_improve = 0 if score.is_best else loop.epochs_no_improve + 1

    # 7. Check governor directives (curriculum fraction/resolution promotion before saving
    # checkpoints)
    audit = (
        _audit_governor(ctx, epoch, score, loop.epochs_no_improve)
        if score.unmeasured is None
        else _AuditOutcome(False, False)
    )

    # 7b. Stochastic Weight Averaging (no-op when disabled; never touches the governor)
    if ctx.governance.swa is not None:
        ctx.governance.swa.end_epoch(ctx, epoch)

    # 8. Checkpoint serialization payload (persists post-audit governor state for next epoch
    # resume)
    _save_checkpoints(ctx, epoch, score, loop.best_metrics)
    # 9. Cloud sync trigger if configured
    maybe_trigger_cloud_sync(ctx, epoch, score.is_best)
    # 10. External telemetry callback
    if _notify_epoch_end(epoch, score, hooks):
        return _cancelled_summary(ctx, loop, epoch)

    loop.last_epoch_executed = epoch
    pillars = _completion_pillars(ctx, snapshot.state, score.combined)
    _apply_budget_decision(ctx, loop, epoch, pillars, audit)
    return None


def _run_epochs(ctx: TrainingContext, loop: _LoopState, hooks: _Hooks) -> TrainingSummary | None:
    """Run epochs until the budget policy stops; return a summary when the run ends early.

    An early end is a user cancellation or the Kaggle session deadline (checked before each
    epoch and, through :class:`SessionDeadlineReached`, inside it).
    """
    epoch = loop.start_epoch
    while loop.stop_reason is None:
        if hooks.cancelled():
            print(f"[CANCEL] Training aborted at epoch {epoch} by user cancellation.", flush=True)
            return _cancelled_summary(ctx, loop, max(0, epoch - 1))
        if _deadline_before_epoch(ctx):
            return _deadline_summary(
                ctx, loop, SessionDeadlineReached(loop.last_epoch_executed, "epoch", 0)
            )
        try:
            cancelled = _run_epoch(ctx, loop, epoch, hooks)
        except SessionDeadlineReached as stop:
            return _deadline_summary(ctx, loop, stop)
        if cancelled is not None:
            return cancelled
        epoch += 1
    return None


def run_training(
    ctx: TrainingContext,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TrainingSummary:
    """Run the complete training loop across all configured epochs.

    Coordinates:
    - Epoch training and validation
    - Governor curriculum and thermal adjustments
    - SOTA tracking and metric vault updates
    - Atomic checkpoint saving for progress and best models
    - SOTA rollback recoil on prolonged plateau or divergence
    - Automated cloud synchronization triggers
    - Optional epoch completion callback invocation
    - Cancellation checks for responsive user termination

    Args:
        ctx: Configured TrainingContext containing model, data, optimizer, and governance.
        on_epoch_end: Optional callback invoked after each epoch with epoch number and metrics.
        cancel_check: Optional callable returning True when cancellation is requested.

    Returns:
        TrainingSummary: Results summary containing final epoch and best metrics.
    """
    start_epoch = _resolve_start_epoch(ctx)
    _prepare_run(ctx, start_epoch)
    hooks = _Hooks(on_epoch_end=on_epoch_end, cancel_check=cancel_check)
    loop = _LoopState(
        start_epoch=start_epoch,
        started_at=time.time(),
        budget_policy=EpochBudgetPolicy(
            total_epochs=ctx.run.total_epochs,
            config=EpochBudgetConfig.from_config(ctx.run.config),
        ),
    )

    early_end = _run_epochs(ctx, loop, hooks)
    if early_end is not None:
        return early_end

    if ctx.governance.swa is not None:
        ctx.governance.swa.finish(ctx, loop.last_epoch_executed)

    # Post-training evaluation on held-out test split if available
    test_metrics: dict[str, float] = {}
    if ctx.data.test_loader is not None and len(ctx.data.test_loader) > 0:
        logger.info("Executing final evaluation pass on held-out test split...")
        test_metrics = validate_one_epoch(
            ctx, epoch=loop.last_epoch_executed, loader=ctx.data.test_loader
        )
        logger.info("Test split evaluation completed: %s", test_metrics)

    total_elapsed = time.time() - loop.started_at
    final_status = "completed" if loop.stop_reason is StopReason.CONVERGED else "partially_trained"

    summary_metrics = dict(loop.best_metrics)
    if test_metrics:
        for k, v in test_metrics.items():
            test_key = f"test_{k}" if not k.startswith("test_") else k
            summary_metrics[test_key] = v

    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=loop.last_epoch_executed,
        best_metrics=summary_metrics,
        total_time=total_elapsed,
        status=final_status,
        stop_reason=loop.stop_reason.value if loop.stop_reason is not None else "",
    )
