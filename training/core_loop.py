"""LemGendary Model Training Suite Universal Trainer (Facade).

Backward-compatible entry point delegating runtime execution, optimization,
AMP management, and validation to the modular training.training subpackage.

YOLOv8n is handled via Ultralytics native trainer (_run_yolo_native).
All other models use the universal TrainingContext path.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import yaml
import torch
import torch.nn as nn

logger = logging.getLogger("lemtrain.core_loop")

from models.factory import get_model
from training.checkpoint import (
    CheckpointRecoveryEngine,
    MetricVault,
    ResumeState,
    safe_load_checkpoint,
)
from training.cloud.git_hub import git_hub_sync
from training.cloud_sync import trigger_cloud_sync
from training.config.secrets import load_secrets
from training.data import (
    build_train_loader,
    build_val_loader as canonical_build_val_loader,
)
from training.export import export_all
from training.governance import SmartTrainingGovernor, SotaTracker
from training.hardware.discovery import discover_device
from training.hardware.policy import apply_hardware_policy
from training.hardware.sentinel import SentinelGuard
from training.losses import CombinedLoss
from training.parallel import build_parallel_strategy
from training.training import (
    TrainingContext,
    TrainingPaths,
    build_optimizer,
    build_scheduler,
    compute_ssim_gpu,
    create_grad_scaler,
    run_training,
)
from training.utils.interrupt import (
    _ACTIVE_PROCESSES,
    cleanup_active_processes,
    install_signal_handlers,
)
from training.utils.logging import install_force_tty
from training.utils.paths import bootstrap_sys_path, get_project_root

bootstrap_sys_path()
install_force_tty()
install_signal_handlers()

load_pat = load_secrets
_active_processes = _ACTIVE_PROCESSES


def build_cli_parser() -> argparse.ArgumentParser:
    """Build unified command-line argument parser for LemGendary Training Suite."""
    parser = argparse.ArgumentParser(description="LemGendary Training Suite Universal Trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="professional_multitask_restoration",
        help="Model key from unified_models.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument(
        "--env",
        type=str,
        default="local",
        choices=["local", "kaggle", "colab"],
        help="Execution environment",
    )
    parser.add_argument(
        "--prefetch_datasets",
        type=str,
        default="",
        help="Comma-separated kaggle endpoints natively executed asynchronously",
    )
    parser.add_argument("--hub_user", type=str, default=None, help="GitHub username for model hub")
    parser.add_argument("--hub_repo", type=str, default=None, help="GitHub repository name for model hub")
    parser.add_argument(
        "--auto_sync",
        action="store_true",
        help="Enable automated cloud synchronization per epoch (Kaggle only)",
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="Bypass loaded scheduler state and re-initialize fresh curve at current step",
    )
    parser.add_argument(
        "--clean",
        "--fresh",
        dest="clean",
        action="store_true",
        help="Start training fresh from epoch 1, wiping local checkpoints and ignoring hub checkpoints",
    )
    parser.add_argument("--phase", type=int, default=1, help="Training Phase (e.g. Pre-training=1, Fine-tuning=2)")
    parser.add_argument("--fold", type=int, default=1, help="Walk-forward fold index (1..6)")
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="List of active pairs for Forex dataset (e.g. EURUSD GBPUSD)",
    )
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=None,
        help="List of active timeframes in minutes (e.g. 60 240 1440)",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Force a specific number of workers")
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=None,
        help="Force a specific number of validation workers",
    )
    parser.add_argument(
        "--enable-batch-growth",
        action="store_true",
        help="Allow intra-epoch physical batch growth when VRAM headroom exceeds 40 percent.",
    )
    parser.add_argument(
        "--parallel",
        choices=["auto", "single", "dp", "ddp"],
        default="auto",
        help="Parallel strategy. 'auto' picks based on device count and model type.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input spatial resolution or target ladder stage in pixels",
    )
    parser.add_argument(
        "--enable-sawtooth",
        action="store_true",
        default=True,
        help="Enable Sawtooth VRAM Governor memory guard",
    )
    return parser


def build_training_context(args: argparse.Namespace) -> TrainingContext:
    """Construct deterministic TrainingContext from parsed CLI arguments and workspace configs."""
    load_secrets()
    project_root = get_project_root()
    config_path = project_root / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    unified_models_rel = config.get("unified_models", "models/unified_models.yaml")
    unified_models_path = project_root / unified_models_rel
    with open(unified_models_path, "r", encoding="utf-8") as f:
        unified_models_registry = yaml.safe_load(f) or {}

    model_key = args.model
    models_dict = unified_models_registry.get("models", {})
    model_info = models_dict.get(model_key, {})

    # 1. Hardware discovery & policy
    device_info = discover_device()
    policy = apply_hardware_policy(model_key, model_info, device_info, config)

    # 2. Directory structure — exclusively under LemGendaryModels
    # Training suite project is code-only; all artifacts, checkpoints, exports, and metrics go to LemGendaryModels
    models_hub_dir = (project_root.parent / "LemGendaryModels" / model_key).resolve()
    models_hub_checkpoint_dir = models_hub_dir / "checkpoints"
    models_hub_dir.mkdir(parents=True, exist_ok=True)
    models_hub_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths = TrainingPaths(
        project_root=project_root,
        local_checkpoint_dir=models_hub_checkpoint_dir,
        hub_checkpoint_dir=models_hub_checkpoint_dir,
        export_dir=models_hub_dir,
        progress_local_path=models_hub_checkpoint_dir / "progress.pth",
        best_checkpoint_path=models_hub_checkpoint_dir / "best.pth",
        history_csv_path=models_hub_dir / "history.csv",
        models_hub_dir=models_hub_dir,
        models_hub_checkpoint_dir=models_hub_checkpoint_dir,
        models_hub_progress_path=models_hub_checkpoint_dir / "progress.pth",
        models_hub_best_path=models_hub_checkpoint_dir / "best.pth",
        models_hub_metrics_csv=models_hub_dir / "metrics.csv",
    )

    if args.clean:
        for clean_target in [
            models_hub_checkpoint_dir / "progress.pth",
            models_hub_checkpoint_dir / "best.pth",
            models_hub_checkpoint_dir / f"{model_key}_latest.pth",
            models_hub_checkpoint_dir / f"{model_key}_best.pth",
            models_hub_dir / "history.csv",
            models_hub_dir / "metrics.csv",
        ]:
            if clean_target.exists():
                try:
                    clean_target.unlink()
                except OSError:
                    pass

    # 3. Model construction
    raw_model = get_model(model_key, model_info)
    raw_model.to(device_info.device)

    # Parallel strategy wrapping — pass model_info so preferred_parallel is honored.
    parallel_strategy = build_parallel_strategy(
        name=args.parallel,
        model=raw_model,
        device=device_info.device,
        model_key=model_key,
        model_info=model_info,
        config=config,
    )
    model = parallel_strategy.setup()

    # 4. Total epochs resolution
    epochs_val = args.epochs or model_info.get("epochs") or config.get("training", {}).get("default_epochs", 100)
    total_epochs = int(epochs_val)

    # 5. Optimizer & Scheduler
    optimizer = build_optimizer(raw_model, config, lr=args.lr)
    scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
        total_epochs=total_epochs,
        reset_scheduler=args.reset_scheduler,
    )

    # 6. Criterion
    raw_task_type = (
        model_info.get("task_type")
        or model_info.get("dataset_type")
        or model_info.get("category")
        or "restoration"
    )
    task_type = raw_task_type[0] if isinstance(raw_task_type, list) else str(raw_task_type)
    criterion = CombinedLoss(task_type=task_type)

    # 7. Data Loaders
    batch_size = args.batch_size or model_info.get("batch_size", 16)
    if task_type == "forex":
        from data.forex_dataset import ForexDataset
        _forex_shard_root = str(
            config.get("paths", {}).get("datasets_root", "../LemGendaryDatasets")
        )
        kwargs = model_info.get("kwargs", {})
        _forex_pairs = (
            getattr(args, "pairs", None)
            or model_info.get("pairs")
            or kwargs.get("pairs")
        )
        _forex_timeframes = (
            getattr(args, "timeframes", None)
            or model_info.get("active_timeframes")
            or kwargs.get("active_timeframes")
        )
        _forex_fold = getattr(args, "fold", 1)
        train_ds = ForexDataset(
            shard_root=_forex_shard_root,
            pairs=_forex_pairs,
            active_timeframes=_forex_timeframes,
            is_train=True,
            fold=_forex_fold,
        )
        try:
            val_ds = ForexDataset(
                shard_root=_forex_shard_root,
                pairs=_forex_pairs,
                active_timeframes=_forex_timeframes,
                is_train=False,
                fold=_forex_fold,
            )
        except Exception as forex_val_err:
            logger.debug("Forex val dataset failed: %s", forex_val_err)
            val_ds = None
    else:
        from data.dataset import MultiTaskDataset
        train_ds = MultiTaskDataset(
            config,
            model_key=model_key,
            is_train=True,
            env=args.env,
            sample_fraction=1.0,
        )
        try:
            val_ds = MultiTaskDataset(
                config,
                model_key=model_key,
                is_train=False,
                env=args.env,
                sample_fraction=1.0,
            )
            if len(val_ds) == 0:
                logger.info("Validation dataset contains 0 samples; running without validation.")
                val_ds = None
        except Exception:
            val_ds = None

    train_loader = build_train_loader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device_info.device,
        env=args.env,
        config=config,
    )
    if val_ds is not None and len(val_ds) > 0:
        try:
            val_loader = canonical_build_val_loader(
                dataset=val_ds,
                batch_size=batch_size,
                num_workers=args.val_num_workers or args.num_workers,
                device=device_info.device,
                env=args.env,
                config=config,
            )
        except Exception:
            val_loader = None
    else:
        val_loader = None

    # 8. Sentinels & Governance
    sentinel = SentinelGuard(device=device_info.device)
    governor = SmartTrainingGovernor(model_info=model_info, config=config)
    sota_tracker = SotaTracker()
    vault = MetricVault()

    # 9. Recovery inspection
    resume_state = None
    if not args.clean:
        recovery_engine = CheckpointRecoveryEngine(
            workspace_root=project_root.parent,
            project_root=project_root,
            env=args.env,
        )
        candidate_roots = recovery_engine.find_candidate_roots(model_key, config=config)
        discovered = recovery_engine.discover_checkpoints(candidate_roots, model_key)
        ckpt_candidate = discovered.get("progress") or discovered.get("latest") or discovered.get("best")
        if ckpt_candidate and ckpt_candidate.exists():
            loaded_data = safe_load_checkpoint(ckpt_candidate, map_location=device_info.device)
            if loaded_data and "model_state" in loaded_data:
                raw_model.load_state_dict(loaded_data["model_state"], strict=False)
                if "optimizer_state" in loaded_data and loaded_data["optimizer_state"]:
                    optimizer.load_state_dict(loaded_data["optimizer_state"])
                resume_state = ResumeState(
                    epoch=loaded_data.get("epoch", 0),
                    iteration=loaded_data.get("iteration", 0),
                    model_state_dict=loaded_data["model_state"],
                    optimizer_state_dict=loaded_data.get("optimizer_state"),
                    scheduler_state_dict=loaded_data.get("scheduler_state"),
                    best_score=loaded_data.get("best_score", 0.0),
                    sota_achieved=loaded_data.get("sota_achieved", False),
                )

    scaler = create_grad_scaler(policy)

    return TrainingContext(
        model_name=model_key,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device_info=device_info,
        policy=policy,
        sentinel=sentinel,
        governor=governor,
        sota_tracker=sota_tracker,
        vault=vault,
        paths=paths,
        config=config,
        model_info=model_info,
        args=args,
        total_epochs=total_epochs,
        accumulation_steps=1,
        scaler=scaler,
        resume_state=resume_state,
        raw_model=raw_model,
        parallel_strategy=parallel_strategy,
    )


def _run_yolo_native(
    args: argparse.Namespace,
    config: dict,
    project_root: Path,
    on_epoch_end: Any = None,
    cancel_check: Any = None,
) -> Any:
    """Delegate yolov8n training to the Governed YOLO Curriculum Runner.

    Coordinates multi-stage resolution ladder progression (320px -> 480px -> 640px),
    hardware-aware Sawtooth VRAM batch allocation, dataset fraction scaling,
    and real-time telemetry synchronization while preserving native Ultralytics
    inner-loop optimizations.
    """
    from training.governance.yolo_governor import run_governed_yolo_training

    return run_governed_yolo_training(
        args=args,
        config=config,
        project_root=project_root,
        on_epoch_end=on_epoch_end,
        cancel_check=cancel_check,
    )


def main(raw_args: list[str] | None = None) -> None:
    """Primary execution entry point for LemGendary Model Training Suite."""
    parser = build_cli_parser()
    args = parser.parse_args(raw_args)

    print("[BOOT] LemGendary Training Suite initiating...", flush=True)

    # Intercept yolov8n before building TrainingContext — it uses the Ultralytics native trainer.
    if args.model == "yolov8n":
        load_secrets()
        project_root = get_project_root()
        config_path = project_root / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        _run_yolo_native(args, config, project_root)
        return

    ctx = build_training_context(args)

    try:
        summary = run_training(ctx)
        print(f"[SUCCESS] Training completed at epoch {summary.final_epoch} with status: {summary.status}")

        # Post-training export
        export_all(
            model_key=ctx.model_name,
            checkpoint_path=ctx.paths.best_checkpoint_path,
            config=ctx.config,
            output_dir=ctx.paths.export_dir,
        )

        # Post-training documentation & notebooks
        try:
            from training.doc_generator import build_model_readme

            readme_text = build_model_readme(
                ctx.model_name,
                ctx.config.get("unified_models", {}),
                summary.final_epoch,
                summary.best_metrics,
            )
            with open(ctx.paths.export_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_text)
        except Exception as doc_err:
            print(f"[WARNING] Model README generation failed: {doc_err}")

        try:
            from training.notebook_generator import (
                generate_colab_inference_notebook,
                generate_colab_usage_notebook,
                generate_inference_notebook,
                generate_usage_notebook,
            )

            generate_inference_notebook(ctx.model_name, str(ctx.paths.export_dir))
            generate_usage_notebook(ctx.model_name, str(ctx.paths.export_dir))
            generate_colab_inference_notebook(ctx.model_name, str(ctx.paths.export_dir))
            generate_colab_usage_notebook(ctx.model_name, str(ctx.paths.export_dir))
        except Exception as nb_err:
            print(f"[WARNING] Notebook generation failed: {nb_err}")

        # Final cloud synchronization
        if args.env == "kaggle" and not getattr(args, "skip_sync", False):
            try:
                trigger_cloud_sync(ctx.model_name, summary.final_epoch, ctx.config)
            except Exception as sync_err:
                print(f"[WARNING] Final cloud sync failed: {sync_err}")

    finally:
        # Always clean up parallel strategy (destroys DDP process group if active).
        if ctx.parallel_strategy is not None:
            try:
                ctx.parallel_strategy.cleanup()
            except Exception as cleanup_err:
                print(f"[WARNING] Parallel strategy cleanup failed: {cleanup_err}")
        cleanup_active_processes()


if __name__ == "__main__":
    main()