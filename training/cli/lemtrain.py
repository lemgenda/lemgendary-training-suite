"""Canonical Typer CLI for LemGendary Model Training Suite.

Dispatches high-level training, evaluation, export, notebook, checkpoint, and audit
operations directly in-process via training.services, or delegates to local sidecar daemon.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Annotated

import typer

from training import model_soup
from training.cli.llm_commands import llm_app
from training.cli.server_commands import server_app
from training.cli.service_calls import announce_sidecar, export_targets, run_training
from training.cli.sync_commands import sync_app
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.eval_service import EvaluationService
from training.services.notebook_service import NotebookService
from training.services.training_service import TrainingService
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.cli.lemtrain")

app = typer.Typer(
    name="lemtrain",
    help="LemGendary Model Training Suite Canonical Command Line Interface.",
    no_args_is_help=True,
)

checkpoints_app = typer.Typer(name="checkpoints", help="Checkpoint management and inspection.")
presets_app = typer.Typer(name="presets", help="Preset profile exploration and management.")
audit_app = typer.Typer(name="audit", help="System resources, model topology, and judicial audits.")
notebooks_app = typer.Typer(
    name="notebooks", help="Jupyter training, inference, and usage notebook generation."
)

app.add_typer(checkpoints_app, name="checkpoints")
app.add_typer(presets_app, name="presets")
app.add_typer(audit_app, name="audit")
app.add_typer(sync_app, name="sync")
app.add_typer(server_app, name="server")
app.add_typer(notebooks_app, name="notebooks")
app.add_typer(llm_app, name="llm")


@app.command("soup")
def soup(
    model: str = typer.Argument(..., help="Model key to blend vault checkpoints for."),
    donors: str | None = typer.Option(
        None,
        "--donors",
        "-d",
        help="Comma-separated vault metrics (psnr,lpips). Default: every vault_*.pth.",
    ),
    weights: str | None = typer.Option(
        None, "--weights", "-w", help="Comma-separated weights (0.6,0.4)"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output filename or path"),
) -> None:
    """Blend specialized vault checkpoints using Pareto Model Soup."""
    parsed_donors = [x.strip() for x in donors.split(",") if x.strip()] if donors else None
    try:
        parsed_weights = [float(x.strip()) for x in weights.split(",")] if weights else None
    except ValueError as exc:
        typer.echo(f"[ERROR] Invalid --weights value '{weights}': {exc}")
        raise typer.Exit(1) from exc

    try:
        result = model_soup.blend_model_soup(
            model,
            project_root=get_project_root(),
            donors=parsed_donors,
            weights=parsed_weights,
            output=output,
        )
    except model_soup.SoupError as exc:
        typer.echo(f"[ERROR] {exc}")
        raise typer.Exit(1) from exc

    typer.echo(f"[INFO] Donors: {', '.join(p.name for p in result.donors)}")
    typer.echo(f"[SUCCESS] Pareto Model Soup created at {result.output_path}")


@app.command("version")
def version() -> None:
    """Show LemGendary Training Suite version."""
    typer.echo("LemGendary Model Training Suite v16.9.102 (Refactored & Hardened)")


@app.command("train")
def train(
    model: str = typer.Argument(..., help="Model architecture key (e.g. nima_aesthetic_mobile)."),
    epochs: int | None = typer.Option(None, "--epochs", "-e", help="Total epochs to train."),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Batch size per step."),
    lr: float | None = typer.Option(None, "--lr", help="Base learning rate."),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Canonical preset profile from presets.yaml."
    ),
    env: str = typer.Option(
        "local", "--env", help="Execution environment ('local', 'kaggle', 'colab')."
    ),
    clean: bool = typer.Option(False, "--clean", help="Wipe local checkpoints and start fresh."),
    auto_sync: bool = typer.Option(
        False, "--auto-sync", help="Enable cloud checkpoint synchronization."
    ),
    parallel: str = typer.Option(
        "auto", "--parallel", help="Parallel execution strategy ('auto', 'single', 'dp', 'ddp')."
    ),
) -> None:
    """Launch model training workflow in-process."""
    announce_sidecar()
    typer.echo(
        f"[START] Initiating training run for model '{model}' (preset={preset}, env={env})..."
    )
    summary = run_training(
        model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        preset=preset,
        env=env,
        clean=clean,
        auto_sync=auto_sync,
        parallel=parallel,
    )

    typer.echo(f"[SUCCESS] Training complete for '{summary.model_name}' (status={summary.status}).")
    typer.echo(f"Final Epoch Completed: {summary.final_epoch}")
    typer.echo(f"Total Elapsed Time: {summary.total_time:.2f}s")
    typer.echo(f"Best Metrics: {summary.best_metrics}")


@app.command("eval")
def evaluate(
    model: str = typer.Argument(..., help="Model architecture key to evaluate."),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", "-c", help="Path to checkpoint file."
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "-b", help="Evaluation batch size."
    ),
    env: str = typer.Option(
        "local", "--env", help="Execution environment ('local', 'kaggle', 'colab')."
    ),
) -> None:
    """Evaluate a trained model checkpoint in-process."""
    typer.echo(f"[EVAL] Evaluating model '{model}'...")
    service = EvaluationService()
    result = service.evaluate(
        model_key=model,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        env=env,
    )

    typer.echo("[RESULTS] Evaluation metrics:")
    for metric_name, val in result.get("metrics", {}).items():
        typer.echo(f"  {metric_name}: {val:.6f}")
    typer.echo(f"  Checkpoint: {result.get('checkpoint_path')}")
    typer.echo(f"  Device: {result.get('device')}")


@app.command("export")
def export(
    model: str = typer.Argument(..., help="Model key to export."),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint", "-c", help="Path to checkpoint file."
    ),
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            "-t",
            help="Target formats ('onnx_fp32', 'onnx_fp16', 'torch_pt', 'webgpu', 'forex').",
        ),
    ] = None,
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Target directory for binaries."
    ),
) -> None:
    """Compile and export model binaries into deployment targets."""
    typer.echo(f"[EXPORT] Compiling export targets for '{model}'...")
    results = export_targets(model, checkpoint, output_dir, target)

    typer.echo("[SUCCESS] Artifacts exported:")
    for tgt, path in results.items():
        typer.echo(f"  [{tgt}] -> {path}")


@app.command("test-all")
def test_all(
    epochs: int = typer.Option(
        1, "--epochs", "-e", help="Epochs per model (default: 1 for unit test)."
    ),
    yes: bool = typer.Option(
        True, "--yes", "-y", help="Auto-confirm all models without interactive prompt."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force execution even if artifacts exist."
    ),
    env: str = typer.Option("local", "--env", help="Execution environment ('local', 'kaggle')."),
) -> None:
    """Run sequential diagnostic unit-test pass across all registered neural models."""
    typer.echo(f"[TEST-ALL] Launching diagnostic fleet test (epochs={epochs}, env={env})...")
    project_root = Path(__file__).resolve().parent.parent.parent
    train_all_script = project_root / "tools" / "train_all.py"
    cmd = [sys.executable, str(train_all_script), "--epochs", str(epochs), "--env", env]
    if yes:
        cmd.append("--yes")
    if force:
        cmd.append("--force")
    code = subprocess.call(cmd, cwd=str(project_root))
    if code != 0:
        raise typer.Exit(code=code)
    typer.echo("[SUCCESS] Fleet diagnostic unit-test pass completed.")


@app.command("notebooks")
def notebooks(
    model: str = typer.Argument(..., help="Model key for notebook generation."),
    platform: str = typer.Option(
        "kaggle", "--platform", "-p", help="Target platform ('kaggle' or 'colab')."
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Directory for generated .ipynb files."
    ),
) -> None:
    """Generate standalone training, inference, and usage notebooks."""
    typer.echo(f"[NOTEBOOKS] Synthesizing {platform} notebooks for '{model}'...")
    service = NotebookService()
    results = service.generate_notebooks(
        model_key=model,
        platform=platform,
        output_dir=output_dir,
    )

    typer.echo("[SUCCESS] Generated notebooks:")
    for kind, path in results.items():
        typer.echo(f"  [{kind}] -> {path}")


@checkpoints_app.command("list")
def checkpoints_list(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Filter checkpoints by model key."
    ),
) -> None:
    """List stored model checkpoints."""
    service = CheckpointService()
    records = service.list_checkpoints(model_key=model)
    if not records:
        typer.echo("No checkpoints found.")
        return

    typer.echo(f"Found {len(records)} checkpoints:")
    for r in records:
        best_tag = "[BEST]" if r["is_best"] else ""
        epoch_tag = f"Epoch {r['epoch']}" if r["epoch"] is not None else "Epoch ?"
        typer.echo(
            f"  [{r['model_key']}] {r['filename']} ({r['size_mb']} MB) {epoch_tag} {best_tag}"
        )


@checkpoints_app.command("inspect")
def checkpoints_inspect(
    path: str = typer.Argument(..., help="Path to checkpoint file to inspect."),
) -> None:
    """Inspect metadata and parameter statistics of a checkpoint."""
    service = CheckpointService()
    info = service.inspect_checkpoint(path)

    typer.echo(f"Checkpoint: {info['filename']}")
    typer.echo(f"Path: {info['path']}")
    typer.echo(f"Size: {info['size_mb']} MB")
    typer.echo(f"Epoch: {info['epoch']}")
    typer.echo(f"Best Loss: {info['best_loss']}")
    typer.echo(f"Parameters: {info['parameter_count']:,}")
    typer.echo(f"Layers Count: {info['layers_count']}")
    typer.echo(f"Top-level Keys: {', '.join(info['keys'])}")


@checkpoints_app.command("prune")
def checkpoints_prune(
    model: str = typer.Argument(..., help="Model key whose checkpoints to prune."),
    keep_last_k: int = typer.Option(
        3, "--keep", "-k", help="Number of recent checkpoints to retain."
    ),
    keep_best: bool = typer.Option(
        True, "--keep-best/--no-keep-best", help="Retain best checkpoint."
    ),
) -> None:
    """Prune intermediate checkpoints to free disk space."""
    service = CheckpointService()
    deleted = service.prune_checkpoints(
        model_key=model, keep_last_k=keep_last_k, keep_best=keep_best
    )
    if deleted:
        typer.echo(f"Pruned {len(deleted)} checkpoints:")
        for name in deleted:
            typer.echo(f"  Deleted: {name}")
    else:
        typer.echo("No checkpoints pruned.")


@presets_app.command("list")
def presets_list() -> None:
    """List canonical training presets from presets.yaml."""
    service = TrainingService()
    presets = service.load_presets()
    if not presets:
        typer.echo("No presets defined in presets.yaml.")
        return

    typer.echo(f"Discovered {len(presets)} canonical presets:")
    for name, cfg in presets.items():
        desc = cfg.get("description", "No description")
        epochs = cfg.get("epochs", "?")
        batch = cfg.get("batch_size", "?")
        lr = cfg.get("learning_rate", "?")
        typer.echo(f"  * {name}: {desc} (epochs={epochs}, batch={batch}, lr={lr})")


@presets_app.command("show")
def presets_show(
    name: str = typer.Argument(..., help="Preset profile name."),
) -> None:
    """Show detailed configuration for a preset."""
    service = TrainingService()
    preset = service.get_preset(name)
    if preset is None:
        typer.echo(f"Preset '{name}' not found.")
        raise typer.Exit(code=1)

    typer.echo(f"Preset: {name}")
    typer.echo(json.dumps(preset, indent=2))


@audit_app.command("system")
def audit_system() -> None:
    """Audit compute hardware, VRAM headroom, and disk capacity."""
    service = AuditService()
    info = service.audit_system()

    typer.echo("System Resource Telemetry:")
    typer.echo(f"  Platform: {info['platform']}")
    typer.echo(f"  Python: {info['python_version']} | PyTorch: {info['torch_version']}")
    typer.echo(f"  Compute Device: {info['device_name']} ({info['device_type']})")
    typer.echo(f"  CUDA Available: {info['cuda_available']} (GPUs: {info['gpu_count']})")
    typer.echo(
        f"  VRAM Total: {info['vram_total_gb']} GB (Allocated: {info['vram_allocated_gb']} GB, "
        f"Reserved: {info['vram_reserved_gb']} GB)"
    )
    typer.echo(f"  CPU Threads: {info['cpu_threads']}")
    typer.echo(f"  Disk Free: {info['disk_free_gb']} GB (Healthy: {info['disk_healthy']})")


@audit_app.command("model")
def audit_model(
    model: str = typer.Argument(..., help="Model key to audit."),
) -> None:
    """Audit model parameter topologies and estimated memory footprint."""
    service = AuditService()
    report = service.audit_model(model)

    typer.echo(f"Model Topology Audit: {report['model_key']} ({report['class_name']})")
    typer.echo(f"  Task Type: {report['task_type']}")
    typer.echo(f"  Target Resolution: {report['target_resolution']}")
    typer.echo(f"  Total Parameters: {report['total_parameters']:,}")
    typer.echo(f"  Trainable Parameters: {report['trainable_parameters']:,}")
    typer.echo(f"  Frozen Parameters: {report['frozen_parameters']:,}")
    typer.echo(f"  Estimated Memory (FP32): {report['estimated_fp32_mb']} MB")
    typer.echo(f"  Estimated Memory (FP16): {report['estimated_fp16_mb']} MB")


@audit_app.command("judicial")
def audit_judicial(
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to .pth or .onnx model."),
    dataset_dir: str = typer.Option(
        ..., "--dataset-dir", "-d", help="Directory of evaluation images."
    ),
    labels_csv: str = typer.Option(..., "--labels-csv", "-l", help="Ground truth labels CSV."),
    output_json: str | None = typer.Option(
        None, "--output-json", "-o", help="Path to save report."
    ),
    model_type: str = typer.Option(
        "nima_aesthetic_mobile", "--model-type", help="Architecture key."
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    device: str = typer.Option("cpu", "--device", help="Execution device ('cuda' or 'cpu')."),
) -> None:
    """Run judicial correlation verification (PLCC / SRCC)."""
    service = AuditService()
    results = service.audit_judicial(
        model_path=model_path,
        dataset_dir=dataset_dir,
        labels_csv=labels_csv,
        output_json=output_json,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
    )

    typer.echo(f"Judicial Correlation Audit Results ({results['samples_evaluated']} samples):")
    typer.echo(f"  Backend: {results['backend']}")
    typer.echo(f"  PLCC: {results['metrics']['PLCC']:.4f}")
    typer.echo(f"  SRCC: {results['metrics']['SRCC']:.4f}")


@notebooks_app.command("list")
def notebooks_list() -> None:
    """List all registered models available for notebook generation."""
    service = NotebookService()
    models = service.list_supported_models()
    typer.echo(f"Discovered {len(models)} models available for notebook generation:")
    for m in models:
        typer.echo(f"  * {m}")


@notebooks_app.command("generate")
def notebooks_generate(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Target model key (e.g. nima_aesthetic_mobile)."
    ),
    platform: str = typer.Option(
        "all", "--platform", "-p", help="Target platform ('kaggle', 'colab', 'all')."
    ),
    kind: Annotated[
        list[str] | None,
        typer.Option(
            "--kind",
            "-k",
            help="Notebook kind ('training', 'inference', 'usage'). Default is all.",
        ),
    ] = None,
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Custom output directory."
    ),
    all_models: bool = typer.Option(
        False, "--all", "-a", help="Generate notebooks for all registered models."
    ),
) -> None:
    """Generate Jupyter training, inference, and usage notebooks."""
    if not model and not all_models:
        typer.echo("[ERROR] Please specify either a model key with --model or use --all.")
        raise typer.Exit(code=1)

    service = NotebookService()
    if all_models:
        typer.echo(f"[START] Generating {platform} notebooks for all registered models...")
        results = service.generate_all_models(platform=platform, kinds=kind, output_dir=output_dir)
        total_generated = sum(len(v) for v in results.values())
        typer.echo(f"[SUCCESS] Generated {total_generated} notebooks across {len(results)} models.")
        for m, files in results.items():
            if files:
                typer.echo(f"  * {m}: {', '.join(files.keys())}")
    else:
        if model is None:
            typer.echo("[ERROR] Model string reference cannot be resolved.")
            raise typer.Exit(code=1)

        typer.echo(f"[START] Generating {platform} notebooks for model '{model}'...")
        try:
            res = service.generate_notebooks(
                model_key=model,
                platform=platform,
                kinds=kind,
                output_dir=output_dir,
            )
            typer.echo(f"[SUCCESS] Generated {len(res)} notebooks for '{model}':")
            for k, p in res.items():
                typer.echo(f"  * {k}: {p}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Notebook generation failed for '%s' (%s): %s", model, platform, e)
            typer.echo(f"[ERROR] Notebook generation failed: {e}")
            raise typer.Exit(code=1) from e


@app.command("test")
def test_unit(
    pattern: str = typer.Option(
        "test_*.py", "--pattern", "-p", help="Test file pattern to execute (e.g. 'test_llm*.py')."
    ),
) -> None:
    """Run unit test suites for the Model Training Suite in-process using unittest."""
    project_root = Path(__file__).resolve().parent.parent.parent
    tests_dir = project_root / "tests" / "unit"
    typer.echo(f"[TEST] Running Training Suite unit tests in {tests_dir} (pattern: {pattern})...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        typer.echo(
            f"[FAIL] Unit tests failed ({len(result.failures)} failures, {len(result.errors)} "
            "errors)"
        )
        raise typer.Exit(1)
    typer.echo(f"[SUCCESS] All {result.testsRun} unit tests passed successfully!")
