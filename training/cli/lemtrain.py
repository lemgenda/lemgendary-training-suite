"""Canonical Typer CLI for LemGendary Model Training Suite.

Dispatches high-level training, evaluation, export, notebook, checkpoint, and audit
operations directly in-process via training.services, or delegates to local sidecar daemon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, List, Optional
import urllib.error
import urllib.request
import typer

from training.server.state import ServerState
from training.services.audit_service import AuditService
from training.services.checkpoint_service import CheckpointService
from training.services.eval_service import EvaluationService
from training.services.export_service import ExportService
from training.services.notebook_service import NotebookService
from training.services.sync_service import CloudSyncService
from training.services.training_service import TrainingService

app = typer.Typer(
    name="lemtrain",
    help="LemGendary Model Training Suite Canonical Command Line Interface.",
    no_args_is_help=True,
)

checkpoints_app = typer.Typer(name="checkpoints", help="Checkpoint management and inspection.")
presets_app = typer.Typer(name="presets", help="Preset profile exploration and management.")
audit_app = typer.Typer(name="audit", help="System resources, model topology, and judicial audits.")
sync_app = typer.Typer(name="sync", help="Cloud storage synchronization and provider telemetry.")
server_app = typer.Typer(name="server", help="FastAPI sidecar daemon process management.")

app.add_typer(checkpoints_app, name="checkpoints")
app.add_typer(presets_app, name="presets")
app.add_typer(audit_app, name="audit")
app.add_typer(sync_app, name="sync")
app.add_typer(server_app, name="server")

SIDECAR_PORT = 8200


def is_sidecar_online(port: int = SIDECAR_PORT) -> bool:
    """Check if the training suite daemon sidecar is listening on port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex(("127.0.0.1", port)) == 0


@app.command("version")
def version() -> None:
    """Show LemGendary Training Suite version."""
    typer.echo("LemGendary Model Training Suite v2026.11.0 (Refactored & Hardened)")


@app.command("train")
def train(
    model: str = typer.Argument(..., help="Model architecture key (e.g. nima_aesthetic_mobile)."),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Total epochs to train."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size per step."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Base learning rate."),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Canonical preset profile from presets.yaml."),
    env: str = typer.Option("local", "--env", help="Execution environment ('local', 'kaggle', 'colab')."),
    clean: bool = typer.Option(False, "--clean", help="Wipe local checkpoints and start fresh."),
    auto_sync: bool = typer.Option(False, "--auto-sync", help="Enable cloud checkpoint synchronization."),
    parallel: str = typer.Option("auto", "--parallel", help="Parallel execution strategy ('auto', 'single', 'dp', 'ddp')."),
) -> None:
    """Launch model training workflow in-process."""
    sidecar_active = is_sidecar_online()
    if sidecar_active:
        typer.echo(f"[INFO] Detected active LemGendary Sidecar Daemon on port {SIDECAR_PORT}.")

    typer.echo(f"[START] Initiating training run for model '{model}' (preset={preset}, env={env})...")

    service = TrainingService()
    summary = service.train(
        model_key=model,
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
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to checkpoint file."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size."),
    env: str = typer.Option("local", "--env", help="Execution environment ('local', 'kaggle', 'colab')."),
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
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to checkpoint file."),
    target: Optional[List[str]] = typer.Option(None, "--target", "-t", help="Target formats ('onnx_fp32', 'onnx_fp16', 'torch_pt', 'webgpu', 'forex')."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Target directory for binaries."),
) -> None:
    """Compile and export model binaries into deployment targets."""
    typer.echo(f"[EXPORT] Compiling export targets for '{model}'...")
    service = ExportService()
    results = service.export(
        model_key=model,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        targets=target,
    )

    typer.echo("[SUCCESS] Artifacts exported:")
    for tgt, path in results.items():
        typer.echo(f"  [{tgt}] -> {path}")


@app.command("notebooks")
def notebooks(
    model: str = typer.Argument(..., help="Model key for notebook generation."),
    platform: str = typer.Option("kaggle", "--platform", "-p", help="Target platform ('kaggle' or 'colab')."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory for generated .ipynb files."),
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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter checkpoints by model key."),
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
        typer.echo(f"  [{r['model_key']}] {r['filename']} ({r['size_mb']} MB) {epoch_tag} {best_tag}")


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
    keep_last_k: int = typer.Option(3, "--keep", "-k", help="Number of recent checkpoints to retain."),
    keep_best: bool = typer.Option(True, "--keep-best/--no-keep-best", help="Retain best checkpoint."),
) -> None:
    """Prune intermediate checkpoints to free disk space."""
    service = CheckpointService()
    deleted = service.prune_checkpoints(model_key=model, keep_last_k=keep_last_k, keep_best=keep_best)
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
    typer.echo(f"  VRAM Total: {info['vram_total_gb']} GB (Allocated: {info['vram_allocated_gb']} GB, Reserved: {info['vram_reserved_gb']} GB)")
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
    dataset_dir: str = typer.Option(..., "--dataset-dir", "-d", help="Directory of evaluation images."),
    labels_csv: str = typer.Option(..., "--labels-csv", "-l", help="Ground truth labels CSV."),
    output_json: Optional[str] = typer.Option(None, "--output-json", "-o", help="Path to save report."),
    model_type: str = typer.Option("nima_aesthetic_mobile", "--model-type", help="Architecture key."),
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


@sync_app.command("probe")
def sync_probe() -> None:
    """Probe health and connectivity across cloud providers."""
    service = CloudSyncService()
    health = service.probe_providers()

    typer.echo("Cloud Provider Health Telemetry:")
    for provider, status in health.items():
        ok = status.get("healthy", False) or status.get("authenticated", False)
        status_label = "HEALTHY" if ok else "UNCONFIGURED"
        typer.echo(f"  [{provider.upper()}] Status: {status_label}")
        for k, v in status.items():
            typer.echo(f"    {k}: {v}")


@sync_app.command("run")
def sync_run(
    model: str = typer.Argument(..., help="Model key to sync."),
    target: str = typer.Option("gdrive", "--target", "-t", help="Target provider ('gdrive', 'kaggle', 'github')."),
    epoch: int = typer.Option(1, "--epoch", "-e", help="Epoch checkpoint to sync."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without transferring files."),
) -> None:
    """Sync model checkpoints and artifacts to remote cloud target."""
    service = CloudSyncService()
    res = service.sync_model(model_key=model, target=target, epoch=epoch, dry_run=dry_run)

    if res.get("success"):
        typer.echo(f"[SUCCESS] Cloud sync complete: {res.get('message', '')}")
    else:
        typer.echo(f"[ERROR] Cloud sync failed: {res.get('message', '')}")


@server_app.command("start")
def server_start(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8200, "--port", "-p", help="Server port"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run detached in background"),
) -> None:
    """Start the FastAPI sidecar daemon."""
    if is_sidecar_online(port):
        typer.echo(f"[INFO] Server is already running on http://{host}:{port}")
        return

    if daemon:
        typer.echo(f"[START] Launching LemGendary Sidecar Daemon on http://{host}:{port} (daemon mode)...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "training.server.app:app", "--host", host, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        state = ServerState()
        state.write_pid(proc.pid)
        token = state.get_or_create_token()
        typer.echo(f"[SUCCESS] Server daemon launched with PID {proc.pid}.")
        typer.echo(f"  Endpoint: http://{host}:{port}/api/health")
        typer.echo(f"  Interactive Docs: http://{host}:{port}/docs")
        typer.echo(f"  Auth Token: {token}")
    else:
        typer.echo(f"[START] Starting LemGendary Sidecar Daemon on http://{host}:{port} (foreground)...")
        import uvicorn
        uvicorn.run("training.server.app:app", host=host, port=port)


@server_app.command("stop")
def server_stop() -> None:
    """Stop running sidecar daemon process."""
    state = ServerState()
    pid = state.read_pid()
    if not pid:
        typer.echo("No active server PID file found.")
        return

    try:
        import psutil
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            p.terminate()
            typer.echo(f"[SUCCESS] Terminated server process with PID {pid}.")
        else:
            typer.echo(f"Process {pid} is no longer running.")
    except Exception as e:
        typer.echo(f"[ERROR] Could not terminate process {pid}: {e}")
    finally:
        state.clear_pid()


@server_app.command("status")
def server_status(
    port: int = typer.Option(8200, "--port", "-p", help="Target port to probe"),
) -> None:
    """Check sidecar daemon status and health endpoint."""
    state = ServerState()
    pid = state.read_pid()
    online = is_sidecar_online(port)

    typer.echo("LemGendary Sidecar Daemon Telemetry:")
    typer.echo(f"  Port: {port}")
    typer.echo(f"  Listening: {'ONLINE' if online else 'OFFLINE'}")
    typer.echo(f"  PID: {pid if pid else 'None'}")
    typer.echo(f"  Token File: {state.token_path}")
    typer.echo(f"  SQLite DB: {state.db_path}")

    if online:
        try:
            url = f"http://127.0.0.1:{port}/api/health"
            req = urllib.request.Request(url, headers={"User-Agent": "LemTrain-CLI"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                typer.echo("  Health Response:")
                typer.echo(f"    Service: {data.get('service')}")
                typer.echo(f"    Version: {data.get('version')}")
                typer.echo(f"    Device: {data.get('device', {}).get('name')}")
        except Exception as e:
            typer.echo(f"  Health probe failed: {e}")


@server_app.command("openapi")
def server_openapi(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="File path to save openapi.json. If omitted, prints to stdout."),
) -> None:
    """Export the OpenAPI 3.1 schema specification of the sidecar daemon."""
    from training.server.app import app as application

    schema = application.openapi()
    content = json.dumps(schema, indent=2) + "\n"
    if output:
        out_path = Path(output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        typer.echo(f"[SUCCESS] Exported OpenAPI specification to {out_path}")
    else:
        typer.echo(content)
