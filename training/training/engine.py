"""Main Training Engine Coordinator for LemGendary Training Suite.

Orchestrates multi-epoch training, validation passes, governance audits,
SOTA tracking, atomic checkpointing, rollback recovery, and cloud sync triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable
import torch

from training.checkpoint import safe_atomic_save
from training.cloud_sync import trigger_cloud_sync
from training.training.context import TrainingContext
from training.training.epoch import train_one_epoch
from training.training.validation import validate_one_epoch


@dataclass(frozen=True)
class TrainingSummary:
    """Summary record produced upon completion of training."""

    model_name: str
    final_epoch: int
    best_metrics: dict[str, float]
    total_time: float
    status: str


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
        start_epoch = ctx.resume_state.epoch + 1

    if ctx.lifecycle_manager is None:
        from training.checkpoint import CheckpointLifecycleManager
        sota_targets = ctx.model_info.get("sota_targets", {})
        ctx.lifecycle_manager = CheckpointLifecycleManager(
            model_name=ctx.model_name,
            project_root=ctx.paths.project_root,
            sota_targets=sota_targets,
        )

    total_start_time = time.time()
    best_metrics: dict[str, float] = {}


    for epoch in range(start_epoch, ctx.total_epochs + 1):
        if cancel_check is not None and cancel_check():
            print(f"[CANCEL] Training aborted at epoch {epoch} by user cancellation.", flush=True)
            return TrainingSummary(
                model_name=ctx.model_name,
                final_epoch=epoch - 1,
                best_metrics=best_metrics,
                total_time=time.time() - total_start_time,
                status="cancelled",
            )

        # 1. Pre-epoch governance step
        if hasattr(ctx.governor, "thermal") and hasattr(ctx.governor.thermal, "step_epoch"):
            ctx.governor.thermal.step_epoch()

        print(f"\n[EPOCH {epoch}/{ctx.total_epochs}] Initiating training pass for {ctx.model_name}...", flush=True)

        # 2. Train one epoch
        train_metrics = train_one_epoch(ctx, epoch, cancel_check=cancel_check)

        if cancel_check is not None and cancel_check():
            print(f"[CANCEL] Training aborted after epoch {epoch} training pass by user cancellation.", flush=True)
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
        current_quality = float(combined_metrics.get("quality_score", max(0.0, 100.0 - current_loss)))
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
            val_info += f" | PSNR: {combined_metrics['psnr']:.2f}dB | SSIM: {combined_metrics.get('ssim', 0.0):.4f}"
        sota_star = " [NEW BEST]" if is_best else ""
        print(
            f"[EPOCH {epoch}/{ctx.total_epochs} COMPLETE] "
            f"TrainLoss: {train_loss:.4f} | "
            f"{val_info} | "
            f"Quality: {current_quality:.2f}{sota_star} | "
            f"EpochTime: {train_metrics.get('epoch_time', 0.0):.1f}s",
            flush=True,
        )

        # 5. Checkpoint serialization payload
        raw_model = ctx.raw_model if ctx.raw_model is not None else ctx.model
        model_state = raw_model.state_dict()

        checkpoint_payload = {
            "epoch": epoch,
            "model_name": ctx.model_name,
            "model_state": model_state,
            "optimizer_state": ctx.optimizer.state_dict(),
            "scheduler_state": ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
            "governor_state": ctx.governor.get_state() if hasattr(ctx.governor, "get_state") else None,
            "best_metrics": best_metrics,
            "current_metrics": combined_metrics,
        }

        # 6. Lifecycle Management: Save latest.pth (and purge progress.pth)
        if ctx.lifecycle_manager is not None:
            ctx.lifecycle_manager.save_latest(epoch=epoch, payload=checkpoint_payload)

            if is_best:
                # Prepare dummy input for ONNX export
                raw_size = ctx.model_info.get("input_size", 256)
                if isinstance(raw_size, list):
                    if len(raw_size) == 3:
                        dummy_shape = (1, int(raw_size[0]), int(raw_size[1]), int(raw_size[2]))
                    elif len(raw_size) == 2:
                        dummy_shape = (1, 3, int(raw_size[0]), int(raw_size[1]))
                    else:
                        dummy_shape = (1, 3, 256, 256)
                elif isinstance(raw_size, int):
                    dummy_shape = (1, 3, raw_size, raw_size)
                else:
                    dummy_shape = (1, 3, 256, 256)
                dummy_input = torch.randn(*dummy_shape, device=ctx.device_info.device)

                ctx.lifecycle_manager.save_best(
                    epoch=epoch,
                    quality_score=current_quality,
                    payload=checkpoint_payload,
                    model=raw_model,
                    dummy_input=dummy_input,
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

        # 7. Vault recording and telemetry export
        ctx.vault.record_epoch(
            epoch=epoch,
            metrics=combined_metrics,
            checkpoint_path=str(ctx.lifecycle_manager.best_path) if (is_best and ctx.lifecycle_manager) else None,
        )
        if ctx.lifecycle_manager is not None:
            ctx.vault.export_csv(ctx.lifecycle_manager.metrics_csv_path)

        # 8. Check governor directives
        if hasattr(ctx.governor, "audit_epoch"):
            try:
                best_q = float(ctx.sota_tracker.best_quality)
                ctx.governor.audit_epoch(
                    current_quality=current_quality,
                    best_quality=best_q,
                    epochs_no_improve=0,
                    regression_epochs=0,
                    current_lr=combined_metrics.get("lr"),
                    current_loss=current_loss,
                    train_loss=train_loss,
                    metrics_dict=combined_metrics,
                )
            except Exception as gov_err:
                print(f"[WARNING] Governor audit skipped: {gov_err}")

        # 9. Cloud sync trigger if configured
        if getattr(ctx.args, "auto_sync", False) and getattr(ctx.args, "env", "local") == "kaggle":
            try:
                trigger_cloud_sync(ctx.model_name, epoch, ctx.config)
            except Exception as sync_err:
                print(f"[WARNING] Automated cloud sync trigger failed: {sync_err}")

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
            except Exception as cb_err:
                print(f"[WARNING] Telemetry epoch callback failed: {cb_err}")

    total_elapsed = time.time() - total_start_time
    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=ctx.total_epochs,
        best_metrics=best_metrics,
        total_time=total_elapsed,
        status="completed",
    )
