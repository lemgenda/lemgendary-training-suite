"""LemGendary Model Training Suite Universal Trainer (Facade).

Backward-compatible entry point delegating runtime execution, optimization,
AMP management, and validation to the modular training.training subpackage.

YOLOv8n is handled via Ultralytics native trainer (run_governed_yolo_training).
All other models use the universal TrainingContext path.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models.factory import get_model
from training.checkpoint import (
    CheckpointLifecycleManager,
    MetricVault,
)
from training.cli_args import add_clean_argument, add_timeframes_argument
from training.cloud_sync import trigger_cloud_sync
from training.config.secrets import load_secrets
from training.context_data import (
    DATASET_BUILD_ERRORS,
    RunSpec,
    build_datasets,
    build_initial_train_loader,
    build_val_loader_for,
    resolve_batch_size,
)
from training.context_resume import LoaderState, TrainState, resume_from_checkpoint
from training.export import export_all
from training.governance import SmartTrainingGovernor, SotaTracker
from training.governance.yolo_governor import run_governed_yolo_training
from training.hardware.discovery import discover_device
from training.hardware.policy import apply_hardware_policy
from training.hardware.sentinel import SentinelGuard
from training.losses import CombinedLoss, ForexDualLoss
from training.model_card import build_model_readme
from training.notebook_generator import (
    generate_colab_inference_notebook,
    generate_colab_usage_notebook,
    generate_inference_notebook,
    generate_usage_notebook,
)
from training.parallel import build_parallel_strategy
from training.training import (
    DataComponents,
    GovernanceComponents,
    HardwareComponents,
    ModelsHubPaths,
    ModelStack,
    RunSettings,
    RunState,
    TrainingContext,
    TrainingPaths,
    build_optimizer,
    build_scheduler,
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

logger = logging.getLogger("lemtrain.core_loop")

bootstrap_sys_path()
install_force_tty()
install_signal_handlers()

load_pat = load_secrets
_active_processes = _ACTIVE_PROCESSES

# Failure modes of post-training artefact generation (docs, notebooks, sync, teardown):
# the same I/O, bad-value and runtime family as dataset construction.
_ARTEFACT_ERRORS: tuple[type[BaseException], ...] = DATASET_BUILD_ERRORS


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
    parser.add_argument(
        "--batch_size",
        type=str,
        default=None,
        help="Batch size per step (positive integer or 'auto')",
    )
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
    parser.add_argument(
        "--hub_repo", type=str, default=None, help="GitHub repository name for model hub"
    )
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
    add_clean_argument(
        parser,
        "Start training fresh from epoch 1, wiping local checkpoints and ignoring hub checkpoints",
    )
    parser.add_argument(
        "--phase", type=int, default=1, help="Training Phase (e.g. Pre-training=1, Fine-tuning=2)"
    )
    parser.add_argument("--fold", type=int, default=1, help="Walk-forward fold index (1..6)")
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="List of active pairs for Forex dataset (e.g. EURUSD GBPUSD)",
    )
    add_timeframes_argument(parser, "List of active timeframes in minutes (e.g. 60 240 1440)")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Force a specific number of workers"
    )
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


def load_workspace_config(project_root: Path) -> dict:
    """Load ``config.yaml`` from the project root."""
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_info(project_root: Path, config: dict, model_key: str) -> dict:
    """Return ``model_key``'s entry in the unified model registry (``{}`` when absent)."""
    unified_models_rel = config.get("unified_models", "unified_models_v2.yaml")
    unified_models_path = project_root / unified_models_rel
    if not unified_models_path.exists():
        unified_models_path = project_root / "unified_models.yaml"
    with open(unified_models_path, encoding="utf-8") as f:
        unified_models_registry = yaml.safe_load(f) or {}

    models_dict = unified_models_registry.get("models")
    if not isinstance(models_dict, dict):
        models_dict = unified_models_registry
    return models_dict.get(model_key, {})


def build_training_paths(project_root: Path, model_key: str) -> TrainingPaths:
    """Create the model's LemGendaryModels directories and return its artefact paths.

    The training suite is code-only; checkpoints, exports and metrics all go to
    ``LemGendaryModels/<model_key>``.
    """
    models_hub_dir = (project_root.parent / "LemGendaryModels" / model_key).resolve()
    models_hub_checkpoint_dir = models_hub_dir / "checkpoints"
    models_hub_dir.mkdir(parents=True, exist_ok=True)
    models_hub_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
        project_root=project_root,
        local_checkpoint_dir=models_hub_checkpoint_dir,
        export_dir=models_hub_dir,
        progress_local_path=models_hub_checkpoint_dir / "progress.pth",
        best_checkpoint_path=models_hub_checkpoint_dir / "best.pth",
        history_csv_path=models_hub_dir / "history.csv",
        models_hub=ModelsHubPaths(
            root=models_hub_dir,
            checkpoint_dir=models_hub_checkpoint_dir,
            progress_path=models_hub_checkpoint_dir / "progress.pth",
            best_path=models_hub_checkpoint_dir / "best.pth",
            metrics_csv=models_hub_dir / "metrics.csv",
        ),
    )


def clean_previous_run(paths: TrainingPaths, model_key: str) -> None:
    """Delete the previous run's checkpoints and metric logs (``--clean``)."""
    hub = paths.models_hub
    if hub is None:
        raise ValueError("clean_previous_run needs the LemGendaryModels paths of the run")
    checkpoint_dir = hub.checkpoint_dir
    hub_dir = hub.root
    for clean_target in [
        checkpoint_dir / "progress.pth",
        checkpoint_dir / "best.pth",
        checkpoint_dir / f"{model_key}_latest.pth",
        checkpoint_dir / f"{model_key}_best.pth",
        hub_dir / "history.csv",
        hub_dir / "metrics.csv",
    ]:
        if clean_target.exists():
            try:
                clean_target.unlink()
            except OSError as unlink_err:
                logger.warning("Clean run could not remove %s: %s", clean_target, unlink_err)


def resolve_task_type(model_info: dict) -> str:
    """The model's task type (first entry of a list), ``restoration`` by default."""
    raw_task_type = (
        model_info.get("task_type")
        or model_info.get("dataset_type")
        or model_info.get("category")
        or "restoration"
    )
    return raw_task_type[0] if isinstance(raw_task_type, list) else str(raw_task_type)


def uses_perceptual_loss(loss_fn_name: str) -> bool:
    """True when the registry ``loss_fn`` names an LPIPS term (e.g. ``l1_lpips``)."""
    return "lpips" in loss_fn_name.lower()


def build_criterion(spec: RunSpec):
    """Forex models train on ``ForexDualLoss``; everything else on ``CombinedLoss``."""
    loss_fn_name = str(spec.model_info.get("loss_fn") or "").lower()
    if (
        loss_fn_name == "forex_dual"
        or spec.task_type in ("forex", "forex_dual")
        or spec.args.model == "forex_predictor"
    ):
        criterion = ForexDualLoss()
    else:
        criterion = CombinedLoss(
            task_type=spec.task_type,
            stabilizers=spec.model_info.get("stabilizers"),
            use_perc=uses_perceptual_loss(loss_fn_name),
        )
    criterion.to(spec.device)
    return criterion


def resolve_total_epochs(args: argparse.Namespace, model_info: dict, config: dict) -> int:
    """CLI epochs, else the model's, else ``training.default_epochs`` (100)."""
    epochs_val = (
        args.epochs
        or model_info.get("epochs")
        or config.get("training", {}).get("default_epochs", 100)
    )
    return int(epochs_val)


def grant_resume_budget(args: argparse.Namespace, resume_state: Any, total_epochs: int) -> int:
    """Extend the epoch budget when the resumed checkpoint has already reached it.

    The run then gets a fresh budget (``--epochs`` or 20) on top of the checkpoint epoch
    instead of ending immediately. Otherwise the configured budget is honoured as-is (no
    hidden 20-epoch floor; see TS D-01).
    """
    if resume_state is not None and resume_state.epoch > 0:
        resumed_ep = resume_state.epoch
        if resumed_ep >= total_epochs:
            additional_epochs = int(args.epochs) if args.epochs else 20
            return resumed_ep + additional_epochs
    return total_epochs


@dataclass
class RunSetup:
    """Per-run services resolved before the model is built."""

    lifecycle_manager: CheckpointLifecycleManager
    device_info: Any
    policy: Any
    paths: TrainingPaths


def resolve_run(args: argparse.Namespace) -> tuple[RunSpec, RunSetup]:
    """Load configs, discover the device, apply the hardware policy and prepare directories."""
    load_secrets()
    project_root = get_project_root()
    config = load_workspace_config(project_root)
    model_key = args.model
    model_info = load_model_info(project_root, config, model_key)

    # Checkpoint lifecycle manager (strict 4-tier governance under LemGendaryModels).
    lifecycle_manager = CheckpointLifecycleManager(
        model_name=model_key,
        project_root=project_root,
        sota_targets=model_info.get("sota_targets", {}),
    )
    device_info = discover_device()
    policy = apply_hardware_policy(model_key, model_info, device_info, config)

    paths = build_training_paths(project_root, model_key)
    if args.clean:
        clean_previous_run(paths, model_key)

    spec = RunSpec(
        args=args,
        config=config,
        model_info=model_info,
        model_key=model_key,
        project_root=project_root,
        task_type=resolve_task_type(model_info),
        device=device_info.device,
    )
    return spec, RunSetup(lifecycle_manager, device_info, policy, paths)


def build_model_stack(spec: RunSpec, total_epochs: int) -> tuple[TrainState, Any]:
    """Build the model, its parallel wrapper, optimizer and scheduler.

    Returns the train state (governor and SOTA tracker still unset) and the parallel strategy.
    """
    raw_model = get_model(spec.model_key, spec.model_info)
    raw_model.to(spec.device)

    # model_info is passed so that preferred_parallel is honored.
    parallel_strategy = build_parallel_strategy(
        name=spec.args.parallel,
        model=raw_model,
        device=spec.device,
        model_key=spec.model_key,
        model_info=spec.model_info,
        config=spec.config,
    )
    model = parallel_strategy.setup()

    optimizer = build_optimizer(raw_model, spec.config, lr=spec.args.lr)
    scheduler = build_scheduler(optimizer=optimizer, config=spec.config, total_epochs=total_epochs)
    train = TrainState(
        raw_model=raw_model,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        governor=None,
        sota_tracker=None,
    )
    return train, parallel_strategy


def build_data(spec: RunSpec, governor: Any, raw_model: Any, batch_size: int):
    """Build the datasets and loaders at the governor's data fraction.

    The governor already resolves the configured fraction (``sample_fraction``,
    ``curriculum.data_fraction``) and the full-data default of LLM / forex models.
    Returns the train ``LoaderState``, validation loader, and optional test loader.
    """
    sample_fraction = getattr(governor.curriculum, "current_fraction", None) or 1.0

    datasets = build_datasets(spec, raw_model, sample_fraction)
    if len(datasets) == 3:
        train_ds, val_ds, test_ds = datasets
    else:
        train_ds, val_ds = datasets
        test_ds = None

    data = LoaderState(
        train_ds=train_ds,
        train_loader=build_initial_train_loader(spec, train_ds, batch_size, sample_fraction),
        batch_size=batch_size,
        sample_fraction=sample_fraction,
    )
    val_loader = build_val_loader_for(spec, val_ds, batch_size)
    test_loader = build_val_loader_for(spec, test_ds, batch_size) if test_ds else None
    return data, val_loader, test_loader


def build_training_context(args: argparse.Namespace) -> TrainingContext:
    """Construct deterministic TrainingContext from parsed CLI arguments and workspace configs."""
    spec, setup = resolve_run(args)

    total_epochs = resolve_total_epochs(args, spec.model_info, spec.config)
    train, parallel_strategy = build_model_stack(spec, total_epochs)
    criterion = build_criterion(spec)
    batch_size = resolve_batch_size(spec, train.raw_model)

    # Sentinels and governance.
    train.governor = SmartTrainingGovernor(model_info=spec.model_info, config=spec.config)
    train.governor.curriculum.update_batch_and_accumulation(batch_size)
    train.sota_tracker = SotaTracker()

    data, val_loader, test_loader = build_data(spec, train.governor, train.raw_model, batch_size)

    # Created before the resume so the checkpoint's loss scale can be restored into it.
    train.scaler = create_grad_scaler(setup.policy)
    resume_state = None
    if not args.clean:
        hub = setup.paths.models_hub
        metrics_csv = hub.metrics_csv if hub is not None else None
        resume_state = resume_from_checkpoint(spec, train, data, metrics_csv)
    total_epochs = grant_resume_budget(args, resume_state, total_epochs)

    return TrainingContext(
        model_name=spec.model_key,
        models=ModelStack(
            model=train.model,
            optimizer=train.optimizer,
            scheduler=train.scheduler,
            criterion=criterion,
            scaler=train.scaler,
            raw_model=train.raw_model,
            parallel_strategy=parallel_strategy,
        ),
        data=DataComponents(
            train_loader=data.train_loader,
            val_loader=val_loader,
            accumulation_steps=max(1, int(getattr(train.governor, "current_acc", 1))),
            test_loader=test_loader,
        ),
        hardware=HardwareComponents(
            device_info=setup.device_info,
            policy=setup.policy,
            sentinel=SentinelGuard(device=spec.device),
        ),
        governance=GovernanceComponents(
            governor=train.governor,
            sota_tracker=train.sota_tracker,
            vault=MetricVault(),
            lifecycle_manager=setup.lifecycle_manager,
        ),
        run=RunSettings(
            paths=setup.paths,
            config=spec.config,
            model_info=spec.model_info,
            args=args,
            total_epochs=total_epochs,
        ),
        state=RunState(
            resume_state=resume_state,
        ),
    )


def main(raw_args: list[str] | None = None) -> None:
    """Primary execution entry point for LemGendary Model Training Suite."""
    parser = build_cli_parser()
    args = parser.parse_args(raw_args)

    if os.environ.get("RANK", "0") == "0":
        print("[BOOT] LemGendary Training Suite initiating...", flush=True)

    # Intercept yolov8n before building TrainingContext: it uses the Ultralytics native trainer.
    if args.model == "yolov8n":
        load_secrets()
        project_root = get_project_root()
        config = load_workspace_config(project_root)
        run_governed_yolo_training(args, config, project_root)
        return

    ctx = build_training_context(args)

    try:
        summary = run_training(ctx)
        print(
            f"[SUCCESS] Training completed at epoch {summary.final_epoch} "
            f"with status: {summary.status}"
        )

        # Post-training export
        export_all(
            model_key=ctx.model_name,
            checkpoint_path=ctx.run.paths.best_checkpoint_path,
            config=ctx.run.config,
            output_dir=ctx.run.paths.export_dir,
        )

        # Post-training documentation & notebooks
        try:
            readme_text = build_model_readme(
                ctx.model_name,
                ctx.run.config.get("unified_models", {}),
                summary.final_epoch,
                summary.best_metrics,
            )
            with open(ctx.run.paths.export_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_text)
        except _ARTEFACT_ERRORS as doc_err:
            logger.warning("Model README generation failed: %s", doc_err)

        try:
            generate_inference_notebook(ctx.model_name, str(ctx.run.paths.export_dir))
            generate_usage_notebook(ctx.model_name, str(ctx.run.paths.export_dir))
            generate_colab_inference_notebook(ctx.model_name, str(ctx.run.paths.export_dir))
            generate_colab_usage_notebook(ctx.model_name, str(ctx.run.paths.export_dir))
        except _ARTEFACT_ERRORS as nb_err:
            logger.warning("Notebook generation failed: %s", nb_err)

        # Final cloud synchronization
        if args.env == "kaggle" and not getattr(args, "skip_sync", False):
            try:
                trigger_cloud_sync(ctx.model_name, summary.final_epoch, ctx.run.config)
            except _ARTEFACT_ERRORS as sync_err:
                logger.warning("Final cloud sync failed: %s", sync_err)

    finally:
        # Always clean up parallel strategy (destroys DDP process group if active).
        if ctx.models.parallel_strategy is not None:
            try:
                ctx.models.parallel_strategy.cleanup()
            except _ARTEFACT_ERRORS as cleanup_err:
                logger.warning("Parallel strategy cleanup failed: %s", cleanup_err)
        cleanup_active_processes()


if __name__ == "__main__":
    main()
