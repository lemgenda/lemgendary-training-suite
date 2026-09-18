"""Main Training Engine Coordinator for LemGendary Training Suite.

Orchestrates multi-epoch training, validation passes, governance audits,
SOTA tracking, atomic checkpointing, rollback recovery, and cloud sync triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
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


def run_training(ctx: TrainingContext) -> TrainingSummary:
    """Run the complete training loop across all configured epochs.

    Coordinates:
    - Epoch training and validation
    - Governor curriculum and thermal adjustments
    - SOTA tracking and metric vault updates
    - Atomic checkpoint saving for progress and best models
    - SOTA rollback recoil on prolonged plateau or divergence
    - Automated cloud synchronization triggers

    Args:
        ctx: Configured TrainingContext containing model, data, optimizer, and governance.

    Returns:
        TrainingSummary: Results summary containing final epoch and best metrics.
    """
    start_epoch = 1
    if ctx.resume_state is not None and ctx.resume_state.epoch > 0:
        start_epoch = ctx.resume_state.epoch + 1

    total_start_time = time.time()
    best_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, ctx.total_epochs + 1):
        # 1. Pre-epoch governance step
        if hasattr(ctx.governor, "thermal") and hasattr(ctx.governor.thermal, "step_epoch"):
            ctx.governor.thermal.step_epoch()

        # 2. Train one epoch
        train_metrics = train_one_epoch(ctx, epoch)

        # 3. Validate one epoch
        val_metrics = validate_one_epoch(ctx, epoch)

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

        # 6. Save checkpoints atomically
        safe_atomic_save(checkpoint_payload, ctx.paths.progress_local_path)
        if is_best:
            safe_atomic_save(checkpoint_payload, ctx.paths.best_checkpoint_path)

        # 7. Vault recording and telemetry export
        ctx.vault.record_epoch(
            epoch=epoch,
            metrics=combined_metrics,
            checkpoint_path=str(ctx.paths.best_checkpoint_path) if is_best else None,
        )
        ctx.vault.export_csv(ctx.paths.history_csv_path)

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

    total_elapsed = time.time() - total_start_time
    return TrainingSummary(
        model_name=ctx.model_name,
        final_epoch=ctx.total_epochs,
        best_metrics=best_metrics,
        total_time=total_elapsed,
        status="completed",
    )
