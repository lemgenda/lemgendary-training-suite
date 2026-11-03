"""Main Training Engine Coordinator for LemGendary Training Suite.

Orchestrates multi-epoch training, validation passes, governance audits,
SOTA tracking, atomic checkpointing, rollback recovery, and cloud sync triggers.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import torch
from torch.utils.data.distributed import DistributedSampler

from data.forex_sampler import RowGroupAwareSampler
from training.checkpoint import CheckpointLifecycleManager, ExportSource
from training.cloud_sync import trigger_cloud_sync
from training.data import rebuild_train_loader
from training.data.loaders import LoaderPlacement, TrainLoaderOptions, WorkerOverrides
from training.epoch_budget import (
    EpochAction,
    EpochBudgetConfig,
    EpochBudgetPolicy,
    StopReason,
)
from training.governance.epoch_audit import EpochSignals
from training.governance.metrics import DEFAULT_METRIC_DIRECTIONS
from training.ratchet import (
    FULL_FRACTION_THRESHOLD,
    executed_rung,
    top_resolution_reached,
)
from training.telemetry import TelemetryEngine
from training.training.context import TrainingContext
from training.training.epoch import train_one_epoch
from training.training.summary import TrainingSummary
from training.training.validation import validate_one_epoch
from training.utils.error_types import (
    CLOUD_SYNC_START_ERRORS,
    EPOCH_CALLBACK_ERRORS,
    STATE_RESTORE_ERRORS,
)

logger = logging.getLogger("lemtrain.engine")


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
    start_epoch = 1
    if ctx.resume_state is not None and ctx.resume_state.epoch > 0:
        if getattr(ctx.resume_state, "is_intra_epoch", False):
            start_epoch = ctx.resume_state.epoch
        else:
            start_epoch = ctx.resume_state.epoch + 1

    # Ensure total_epochs is never lower than start_epoch
    if ctx.total_epochs < start_epoch:
        ctx.total_epochs = max(30, start_epoch + 10)

    if ctx.lifecycle_manager is None:
        sota_targets = ctx.model_info.get("sota_targets", {})
        ctx.lifecycle_manager = CheckpointLifecycleManager(
            model_name=ctx.model_name,
            project_root=ctx.paths.project_root,
            sota_targets=sota_targets,
        )

    total_start_time = time.time()
    best_metrics: dict[str, float] = {}
    epochs_no_improve = 0

    epoch = start_epoch
    last_epoch_executed = 0
    budget_policy = EpochBudgetPolicy(
        total_epochs=ctx.total_epochs,
        config=EpochBudgetConfig.from_config(ctx.config),
    )
    stop_reason: StopReason | None = None

    while stop_reason is None:
        if cancel_check is not None and cancel_check():
            print(f"[CANCEL] Training aborted at epoch {epoch} by user cancellation.", flush=True)
            return TrainingSummary(
                model_name=ctx.model_name,
                final_epoch=max(0, epoch - 1),
                best_metrics=best_metrics,
                total_time=time.time() - total_start_time,
                status="cancelled",
            )

        # 1. Pre-epoch governance step
        if hasattr(ctx.governor, "thermal") and hasattr(ctx.governor.thermal, "step_epoch"):
            ctx.governor.thermal.step_epoch()

        is_resuming_val = (
            ctx.resume_state is not None
            and ctx.resume_state.is_intra_epoch
            and ctx.resume_state.phase == "val"
            and epoch == start_epoch
            and not ctx.resumed_val_handled
        )

        if is_resuming_val:
            ctx.resumed_val_handled = True
            print(
                f"\n[RESUME] Resuming directly into validation pass for {ctx.model_name} at Epoch "
                f"{epoch} "
                "(progress checkpoint captured during validation phase; skipping redundant "
                "training pass)...",
                flush=True,
            )
            resume_tm = ctx.resume_state.train_metrics if ctx.resume_state is not None else {}
            train_metrics = dict(resume_tm) if resume_tm else {}
            if "train_loss" not in train_metrics:
                train_metrics["train_loss"] = float(train_metrics.get("loss", 0.0))
            if "epoch_time" not in train_metrics:
                train_metrics["epoch_time"] = 0.0
        else:
            if os.environ.get("RANK", "0") == "0":
                print(
                    f"\n[EPOCH {epoch}/{ctx.total_epochs}] Initiating training pass for "
                    f"{ctx.model_name}...",
                    flush=True,
                )

            # Reseed sampler epoch if supported (e.g. RowGroupAwareSampler)
            sampler = getattr(ctx.train_loader, "sampler", None)
            set_epoch_fn = getattr(sampler, "set_epoch", None)
            if callable(set_epoch_fn):
                set_epoch_fn(epoch)

            # 2. Train one epoch
            train_metrics = train_one_epoch(ctx, epoch, cancel_check=cancel_check)

        # Cache last train metrics on context for validation-phase progress saves
        ctx.last_train_metrics = train_metrics

        if cancel_check is not None and cancel_check():
            print(
                f"[CANCEL] Training aborted after epoch {epoch} training pass by user "
                "cancellation.",
                flush=True,
            )
            return TrainingSummary(
                model_name=ctx.model_name,
                final_epoch=epoch,
                best_metrics=best_metrics,
                total_time=time.time() - total_start_time,
                status="cancelled",
            )

        # 3. Validate one epoch
        val_metrics = validate_one_epoch(ctx, epoch, cancel_check=cancel_check)

        # Combined metrics for governance and telemetry
        combined_metrics = {**train_metrics, **val_metrics}

        # 4. SOTA evaluation
        current_loss = float(combined_metrics.get("val_loss", 0.0))
        train_loss = float(combined_metrics.get("train_loss", 0.0))

        is_forex = (
            ctx.model_info.get("category") == "forex"
            or ctx.model_info.get("task_type") == "forex"
            or ctx.model_name == "forex_predictor"
        )
        task_type_str = "forex" if is_forex else str(ctx.model_info.get("dataset_type", "image"))
        sota_targets = ctx.model_info.get("sota_targets", {})

        if "quality_score" in combined_metrics:
            current_quality = float(combined_metrics["quality_score"])
        elif sota_targets:
            dummy_telemetry = TelemetryEngine(".", task_type="forex" if is_forex else "image")
            computed_qs, _ = dummy_telemetry.compute_quality_score(
                combined_metrics, sota_targets, task_type_str
            )
            current_quality = float(computed_qs)
            combined_metrics["quality_score"] = current_quality
        else:
            current_quality = max(0.0, 100.0 - current_loss)
            combined_metrics["quality_score"] = current_quality

        is_best = ctx.sota_tracker.record_epoch(
            current_quality=current_quality,
            current_loss=current_loss,
            train_loss=train_loss,
        )
        if is_best:
            best_metrics = dict(combined_metrics)

        # Epoch summary logging
        val_info = f"ValLoss: {current_loss:.4f}"
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
        sota_star = " [NEW BEST]" if is_best else ""
        print(
            f"[EPOCH {epoch}/{ctx.total_epochs} COMPLETE] "
            f"TrainLoss: {train_loss:.4f} | "
            f"{val_info} | "
            f"Quality: {current_quality:.2f}{sota_star} | "
            f"EpochTime: {train_metrics.get('epoch_time', 0.0):.1f}s",
            flush=True,
        )

        # 5. Capture active governor state used during this epoch for telemetry and completion audit
        pre_audit_gov_state = ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else {}
        pre_audit_stress = float(
            getattr(ctx.governor, "current_stress", getattr(ctx.governor, "stress", 0.0))
        )

        # 6. Vault recording and canonical telemetry export (records active epoch parameters)
        ctx.vault.record_epoch(
            epoch=epoch,
            metrics=combined_metrics,
            checkpoint_path=str(ctx.lifecycle_manager.best_path)
            if (is_best and ctx.lifecycle_manager)
            else None,
        )
        if ctx.paths.history_csv_path is not None:
            ctx.vault.export_csv(ctx.paths.history_csv_path)

        # Standard LemGendary Telemetry Engine export (metrics.csv)
        if ctx.lifecycle_manager is not None:
            is_forex = (
                ctx.model_info.get("category") == "forex"
                or ctx.model_info.get("task_type") == "forex"
                or ctx.model_name == "forex_predictor"
            )
            task_type = "forex" if is_forex else "image"
            export_dir = str(ctx.lifecycle_manager.model_hub_dir)
            telemetry = TelemetryEngine(export_dir=export_dir, task_type=task_type)
            quality_score = float(combined_metrics.get("quality_score", current_quality))
            fold = int(getattr(ctx.args, "fold", 1) or 1)
            phase = int(getattr(ctx.args, "phase", 1) or 1)
            pairs_cfg = ctx.model_info.get("kwargs", {}).get("pairs", [])
            num_pairs = len(pairs_cfg) if pairs_cfg else 16
            telemetry.write_epoch_row(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=current_loss,
                lr=float(combined_metrics.get("lr", 0.0)),
                curr_metrics=combined_metrics,
                quality_score=quality_score,
                governor_state=pre_audit_gov_state,
                stress=pre_audit_stress,
                num_pairs=num_pairs,
                phase=phase,
                fold=fold,
            )

        if is_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 7. Check governor directives (curriculum fraction/resolution promotion before saving
        # checkpoints)
        governor_early_stop = False
        governor_reports_improvement = False
        if hasattr(ctx.governor, "audit_epoch"):
            try:
                f_chg, r_chg, lr_chg, _t_chg, _c_chg, _b_chg, stop_trig, gov_msg = (
                    ctx.governor.audit_epoch(
                        EpochSignals(
                            current_quality=current_quality,
                            current_loss=current_loss,
                            train_loss=train_loss,
                            metrics_dict=combined_metrics,
                        ),
                        epochs_no_improve=epochs_no_improve,
                    )
                )
                if gov_msg:
                    print(f"{gov_msg}", flush=True)
                governor_early_stop = bool(stop_trig)
                governor_reports_improvement = not governor_early_stop

                # Dynamic manifold scaling: update dataset and rebuild train_loader if fraction,
                # resolution, or batch changed
                if (f_chg or r_chg or _b_chg) and ctx.train_loader is not None:
                    train_ds = getattr(ctx.train_loader, "dataset", None)
                    if train_ds is not None and hasattr(train_ds, "update_strategy"):
                        new_res = getattr(ctx.governor, "current_res", None)
                        new_frac = getattr(ctx.governor, "current_fraction", None)
                        stress_val = getattr(ctx.governor, "current_stress", None)
                        train_ds.update_strategy(
                            size=new_res if r_chg else None,
                            fraction=new_frac if f_chg else None,
                            stress=stress_val,
                        )
                        fallback_batch = ctx.train_loader.batch_size or 16
                        curr_batch = int(
                            getattr(ctx.governor.curriculum, "current_batch", fallback_batch)
                        )
                        ctx.accumulation_steps = max(
                            1, int(getattr(ctx.governor, "current_acc", ctx.accumulation_steps))
                        )
                        curr_sampler = getattr(ctx.train_loader, "sampler", None)
                        if (
                            curr_sampler is not None
                            and type(curr_sampler).__name__ == "RowGroupAwareSampler"
                        ):
                            curr_sampler = RowGroupAwareSampler(train_ds, shuffle=True, seed=42)
                        elif (
                            curr_sampler is not None
                            and type(curr_sampler).__name__ == "DistributedSampler"
                        ):
                            curr_sampler = DistributedSampler(train_ds, shuffle=True)
                        elif curr_sampler is not None and type(curr_sampler).__name__ in (
                            "RandomSampler",
                            "SequentialSampler",
                        ):
                            curr_sampler = None
                        ctx.train_loader = rebuild_train_loader(
                            old_loader=ctx.train_loader,
                            dataset=train_ds,
                            batch_size=curr_batch,
                            options=TrainLoaderOptions(
                                sampler=curr_sampler,
                                shuffle=(curr_sampler is None),
                                workers=WorkerOverrides(
                                    num_workers=ctx.args.num_workers
                                    if hasattr(ctx.args, "num_workers")
                                    else None
                                ),
                            ),
                            placement=LoaderPlacement(
                                device=ctx.device_info.device,
                                env=getattr(ctx.args, "env", "local"),
                                config=ctx.config,
                            ),
                        )
                        print(
                            f"[CURRICULUM] Updated train loader: {len(train_ds)} active samples "
                            f"(fraction={float(new_frac or 1.0) * 100:.0f}%, res={new_res}, "
                            f"batch_size={curr_batch}, total_batches={len(ctx.train_loader)})",
                            flush=True,
                        )
                        logger.info(
                            "Rebuilt train_loader on governor directive: res=%s, fraction=%s, "
                            "batch_size=%d",
                            str(new_res),
                            str(new_frac),
                            curr_batch,
                        )

                # Learning rate propulsion or cooling adjustments
                if lr_chg and ctx.optimizer is not None:
                    lr_mult = float(getattr(ctx.governor, "lr_multiplier", 1.0))
                    for param_group in ctx.optimizer.param_groups:
                        if "initial_lr" in param_group:
                            param_group["lr"] = param_group["initial_lr"] * lr_mult
                        else:
                            param_group["lr"] = param_group.get("lr", 1e-4) * lr_mult
                    logger.debug("Applied governor learning rate multiplier: %.3fx", lr_mult)

            except STATE_RESTORE_ERRORS as gov_err:
                logger.warning(
                    "Governor audit or loader rebuild failed at epoch %d: %s", epoch, gov_err
                )

        # 8. Checkpoint serialization payload (persists post-audit governor state for next epoch
        # resume)
        raw_model = ctx.raw_model if ctx.raw_model is not None else ctx.model
        model_state = raw_model.state_dict()

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": ctx.model_name,
            "model_state": model_state,
            "optimizer_state": ctx.optimizer.state_dict(),
            "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
            "governor_state": ctx.governor.get_state()
            if hasattr(ctx.governor, "get_state")
            else None,
            "best_quality_score": float(ctx.sota_tracker.best_quality),
            "best_metrics": best_metrics,
            "current_metrics": combined_metrics,
        }

        if ctx.lifecycle_manager is not None:
            ctx.lifecycle_manager.save_latest(epoch=epoch, payload=checkpoint_payload)

            if is_best:
                # Prepare dummy input for ONNX export
                raw_size = ctx.model_info.get("input_size")
                if isinstance(raw_size, list):
                    if len(raw_size) == 3:
                        dummy_shape: tuple[int, ...] = (
                            1,
                            int(raw_size[0]),
                            int(raw_size[1]),
                            int(raw_size[2]),
                        )
                    elif len(raw_size) == 2:
                        dummy_shape = (1, 3, int(raw_size[0]), int(raw_size[1]))
                    elif len(raw_size) == 1:
                        dummy_shape = (1, int(raw_size[0]))
                    else:
                        dummy_shape = (1, 3, 256, 256)
                elif isinstance(raw_size, int):
                    dummy_shape = (1, 3, raw_size, raw_size)
                elif (
                    ctx.train_loader is not None
                    and hasattr(ctx.train_loader, "dataset")
                    and hasattr(ctx.train_loader.dataset, "tensors")
                    and len(getattr(ctx.train_loader.dataset, "tensors", ())) > 0
                ):
                    sample_tensor = ctx.train_loader.dataset.tensors[0]
                    dummy_shape = (1, *[int(d) for d in sample_tensor.shape[1:]])
                else:
                    dummy_shape = (1, 3, 256, 256)
                dummy_input = torch.randn(*dummy_shape, device=ctx.device_info.device)

                ctx.lifecycle_manager.save_best(
                    epoch=epoch,
                    quality_score=current_quality,
                    payload=checkpoint_payload,
                    export=ExportSource(model=raw_model, dummy_input=dummy_input),
                )

            # Check and save vault milestones
            ctx.lifecycle_manager.check_and_save_vault(
                epoch=epoch,
                metrics=combined_metrics,
                payload=checkpoint_payload,
            )
        else:
            if is_best and ctx.paths.best_checkpoint_path is not None:
                torch.save(checkpoint_payload, ctx.paths.best_checkpoint_path)
            if ctx.paths.history_csv_path is not None:
                ctx.vault.export_csv(ctx.paths.history_csv_path)

        # 9. Cloud sync trigger if configured
        if getattr(ctx.args, "auto_sync", False) and getattr(ctx.args, "env", "local") == "kaggle":
            try:
                trigger_cloud_sync(ctx.model_name, epoch, ctx.config)
            except CLOUD_SYNC_START_ERRORS as sync_err:
                logger.warning(
                    "Automated cloud sync trigger failed at epoch %d: %s", epoch, sync_err
                )

        # 10. External telemetry callback
        if on_epoch_end is not None:
            try:
                on_epoch_end(epoch, combined_metrics)
            except (InterruptedError, KeyboardInterrupt):
                print(f"[CANCEL] Epoch {epoch} interrupted by cancellation callback.", flush=True)
                return TrainingSummary(
                    model_name=ctx.model_name,
                    final_epoch=epoch,
                    best_metrics=best_metrics,
                    total_time=time.time() - total_start_time,
                    status="cancelled",
                )
            except EPOCH_CALLBACK_ERRORS:
                # Boundary: on_epoch_end is caller-supplied code; an expected failure is logged
                # with traceback and must not abort training.
                logger.exception("Telemetry epoch callback failed at epoch %d", epoch)

        last_epoch_executed = epoch

        # Evaluate 4-Pillar Completion Invariant:
        # 1. Minimum 20 epochs
        # 2. Top resolution ladder rung (or Fold 6 for forex)
        # 3. 100% data fraction (>= 0.999) during the completed epoch
        # 4. All defined SOTA targets met
        is_forex = (
            ctx.model_info.get("category") == "forex"
            or ctx.model_info.get("task_type") == "forex"
            or ctx.model_name == "forex_predictor"
        )
        opt_cfg = ctx.model_info.get("optimization", {})
        # Read the rung this epoch actually trained at (pre-audit state, the
        # keys get_state() writes). Never fall back to the post-audit governor:
        # a spatial jump during the audit must not count as training there.
        executed_res, executed_frac = executed_rung(pre_audit_gov_state)
        if is_forex:
            res_ladder = (
                opt_cfg.get("res_ladder")
                or getattr(ctx.governor, "res_ladder", None)
                or [1, 5, 15, 60, 240, 1440]
            )
            target_res = max(res_ladder) if res_ladder else 1440
            ladder_passed = getattr(ctx.args, "fold", 1) >= 6 or executed_res >= target_res
        else:
            res_ladder = (
                opt_cfg.get("res_ladder")
                or getattr(ctx.governor, "res_ladder", None)
                or [256, 384, 512]
            )
            ladder_passed = top_resolution_reached(executed_res, res_ladder)

        data_fraction_passed = executed_frac >= FULL_FRACTION_THRESHOLD

        sota_targets = ctx.model_info.get("sota_targets", {})
        sota_all_met = True
        if isinstance(sota_targets, dict) and sota_targets:
            for s_key, s_target in sota_targets.items():
                if s_key not in combined_metrics:
                    sota_all_met = False
                    break
                achieved = float(combined_metrics[s_key])
                target_val = float(s_target)
                is_higher = DEFAULT_METRIC_DIRECTIONS.get(s_key, True)
                if is_higher and achieved < target_val or not is_higher and achieved > target_val:
                    sota_all_met = False
                    break

        is_fully_completed = (
            last_epoch_executed >= 20 and ladder_passed and data_fraction_passed and sota_all_met
        )

        decision = budget_policy.decide(
            epoch,
            converged=is_fully_completed,
            early_stop=governor_early_stop,
            governor_reports_improvement=governor_reports_improvement,
        )
        ctx.total_epochs = decision.total_epochs
        if decision.action is EpochAction.STOP:
            stop_reason = decision.stop_reason
            reason_label = stop_reason.value if stop_reason is not None else "stop"
            print(
                f"\n[STOP] [{reason_label}] "
                f"{ctx.model_name}: {decision.message} (Epochs>=20: {last_epoch_executed >= 20}, "
                f"Ladder: {ladder_passed}, Data: {data_fraction_passed}, SOTA: {sota_all_met}).",
                flush=True,
            )
            logger.info("Training loop stopped for %s: %s", ctx.model_name, decision.message)
        elif decision.action is EpochAction.EXTEND:
            print(
                f"\n[GOVERNOR] [BUDGET EXTENSION] {ctx.model_name}: {decision.message} "
                f"(Epochs>=20: {last_epoch_executed >= 20}, Ladder: {ladder_passed}, "
                f"Data: {data_fraction_passed}, SOTA: {sota_all_met}).",
                flush=True,
            )
            logger.info("Epoch budget extended for %s: %s", ctx.model_name, decision.message)

        epoch += 1

    total_elapsed = time.time() - total_start_time
    final_status = "completed" if stop_reason is StopReason.CONVERGED else "partially_trained"

    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=last_epoch_executed,
        best_metrics=best_metrics,
        total_time=total_elapsed,
        status=final_status,
        stop_reason=stop_reason.value if stop_reason is not None else "",
    )
