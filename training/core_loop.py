"""LemGendary Model Training Suite Universal Trainer (Facade).

Backward-compatible entry point delegating runtime execution, optimization,
AMP management, and validation to the modular training.training subpackage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import yaml
import torch
import torch.nn as nn

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

    # 2. Directory structure
    local_checkpoint_dir = project_root / "checkpoints" / model_key
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hub_checkpoint_dir = project_root / "hub" / model_key
    export_dir = project_root / "export" / model_key
    export_dir.mkdir(parents=True, exist_ok=True)

    paths = TrainingPaths(
        project_root=project_root,
        local_checkpoint_dir=local_checkpoint_dir,
        hub_checkpoint_dir=hub_checkpoint_dir,
        export_dir=export_dir,
        progress_local_path=local_checkpoint_dir / "progress.pth",
        best_checkpoint_path=local_checkpoint_dir / "best.pth",
        history_csv_path=local_checkpoint_dir / "history.csv",
    )

    # 3. Model construction
    raw_model = get_model(model_key, model_info)
    raw_model.to(device_info.device)

    # Parallel strategy wrapping
    parallel_strategy = build_parallel_strategy(
        name=args.parallel,
        model=raw_model,
        device=device_info.device,
        model_key=model_key,
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
    task_type = model_info.get("task_type", "restoration")
    criterion = CombinedLoss(task_type=task_type)

    # 7. Data Loaders
    batch_size = args.batch_size or model_info.get("batch_size", 16)
    if task_type == "forex":
        from data.forex_dataset import ForexDataset
        train_ds = ForexDataset(config, is_train=True, env=args.env)
        try:
            val_ds = ForexDataset(config, is_train=False, env=args.env)
        except Exception:
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
    if val_ds is not None:
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
        recovery_engine = CheckpointRecoveryEngine(model_key=model_key)
        ckpt_candidate = recovery_engine.find_best_checkpoint()
        if ckpt_candidate and ckpt_candidate.exists():
            loaded_data = safe_load_checkpoint(ckpt_candidate, map_location=device_info.device)
            if loaded_data and "model_state" in loaded_data:
                raw_model.load_state_dict(loaded_data["model_state"], strict=False)
                if "optimizer_state" in loaded_data and loaded_data["optimizer_state"]:
                    optimizer.load_state_dict(loaded_data["optimizer_state"])
                resume_state = ResumeState(
                    epoch=loaded_data.get("epoch", 0),
                    iteration=loaded_data.get("iteration", 0),
                    best_loss=loaded_data.get("best_loss", float("inf")),
                    best_score=loaded_data.get("best_score", 0.0),
                    checkpoint_path=ckpt_candidate,
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
    )


def main(raw_args: list[str] | None = None) -> None:
    """Primary execution entry point for LemGendary Model Training Suite."""
    parser = build_cli_parser()
    args = parser.parse_args(raw_args)

    print("[BOOT] LemGendary Training Suite initiating...", flush=True)
    ctx = build_training_context(args)

    try:
        summary = run_training(ctx)
        print(f"[SUCCESS] Training completed at epoch {summary.final_epoch} with status: {summary.status}")

        # Post-training export
        raw_model = ctx.raw_model if ctx.raw_model is not None else ctx.model
        export_all(
            model_name=ctx.model_name,
            model=raw_model,
            export_dir=ctx.paths.export_dir,
            checkpoint_path=ctx.paths.best_checkpoint_path,
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

            generate_inference_notebook(ctx.model_name, ctx.paths.export_dir)
            generate_usage_notebook(ctx.model_name, ctx.paths.export_dir)
            generate_colab_inference_notebook(ctx.model_name, ctx.paths.export_dir)
            generate_colab_usage_notebook(ctx.model_name, ctx.paths.export_dir)
        except Exception as nb_err:
            print(f"[WARNING] Notebook generation failed: {nb_err}")

        # Final cloud synchronization
        if args.env == "kaggle" and not getattr(args, "skip_sync", False):
            try:
                trigger_cloud_sync(ctx.model_name, summary.final_epoch, ctx.config)
            except Exception as sync_err:
                print(f"[WARNING] Final cloud sync failed: {sync_err}")

    finally:
        cleanup_active_processes()


if __name__ == "__main__":
    main()