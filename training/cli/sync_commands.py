"""lemtrain sync: cloud storage probe, sync run, model push and pull."""

from __future__ import annotations

import logging

import typer
import yaml

from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.variations import VARIATION_BY_KEY
from training.export.staging import VariationStagingService
from training.registry import get_model_registry
from training.services.sync_service import CloudSyncService

logger = logging.getLogger("lemtrain.cli.lemtrain")


sync_app = typer.Typer(name="sync", help="Cloud storage synchronization and provider telemetry.")


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
    target: str = typer.Option(
        "gdrive", "--target", "-t", help="Target provider ('gdrive', 'kaggle', 'github')."
    ),
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


@sync_app.command("push")
def sync_push(
    model: str | None = typer.Argument(
        None, help="Model key to upload to Kaggle Models (optional if --all is specified)."
    ),
    all_models: bool = typer.Option(
        False, "--all", "-a", help="Push all 22 registered models to Kaggle Models Hub."
    ),
    variation: str = typer.Option(
        "all",
        "--variation",
        "-v",
        help="Variation to push ('all', 'pytorch', 'onnx_fp32', 'onnx_fp16').",
    ),
    version_notes: str | None = typer.Option(
        None, "--notes", "-n", help="Optional release version notes."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate isolation and metadata without uploading to Kaggle."
    ),
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        "-s",
        help="Optional LemGendaryModels root to stage and push from (default: sibling of repo).",
    ),
    username: str | None = typer.Option(
        None, "--username", "-u", help="Optional Kaggle username override."
    ),
) -> None:
    """Stage and upload local model checkpoints, variations, and metrics to Kaggle Models."""
    if not model and not all_models:
        typer.echo("[ERROR] Please specify either a model key or --all.")
        raise typer.Exit(code=1)

    sync_mgr = KaggleModelSyncManager(username=username, hub_root=source_dir)
    staging_service = VariationStagingService()
    # Kaggle/transport/file errors, metadata YAML errors, and RuntimeError (kagglehub missing).
    push_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), yaml.YAMLError, RuntimeError)

    if all_models:
        reg = get_model_registry()
        models = reg.list_models()
        total_models = len(models)
        typer.echo(
            f"[START] Commencing fleet Kaggle push for all {total_models} models (variation: "
            f"{variation}, dry_run: {dry_run})..."
        )
        succeeded = 0
        failed = 0

        for idx, m_spec in enumerate(models, 1):
            m_key = m_spec.key
            typer.echo(f"[{idx}/{total_models}] Processing '{m_key}' ({m_spec.name})...")
            try:
                sync_mgr.ensure_model_metadata(m_key)
                staged = staging_service.stage_variations(
                    m_key, base_dir=sync_mgr.get_model_dir(m_key)
                )
                if staged.staged_files:
                    typer.echo(f"  * Staged {len(staged.staged_files)} isolated variation files.")
                res = sync_mgr.push_model(
                    model_key=m_key,
                    variation=variation,
                    version_notes=version_notes,
                    dry_run=dry_run,
                )
                if res.success:
                    succeeded += 1
                    handles = [v.handle for v in res.variations.values() if v.success]
                    typer.echo(f"  [SUCCESS] Pushed -> {handles}")
                else:
                    failed += 1
                    typer.echo(f"  [WARN] Failed/skipped: {'; '.join(res.errors)}")
            except push_errors as e:
                # Fleet boundary: one model's push failure must not abort the remaining models.
                failed += 1
                logger.exception("Fleet Kaggle push failed for model '%s'", m_key)
                typer.echo(f"  [ERROR] Exception for '{m_key}': {e}")

        typer.echo(
            f"[COMPLETED] Fleet push finished: {succeeded} succeeded, {failed} failed/skipped out "
            f"of {total_models} models."
        )
        if failed > 0 and not dry_run and succeeded == 0:
            raise typer.Exit(code=1)
    else:
        if not model:
            typer.echo(
                "[ERROR] No target model specified. Provide a --model argument or use "
                "--all-models.",
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(
            f"[START] Staging and pushing '{model}' to Kaggle Models (variation: {variation}, "
            f"dry_run: {dry_run})..."
        )
        sync_mgr.ensure_model_metadata(model)
        staged = staging_service.stage_variations(model, base_dir=sync_mgr.get_model_dir(model))
        if staged.staged_files:
            typer.echo(f"  * Staged {len(staged.staged_files)} isolated variation files.")
        res = sync_mgr.push_model(
            model_key=model,
            variation=variation,
            version_notes=version_notes,
            dry_run=dry_run,
        )
        if res.success:
            handles = [v.handle for v in res.variations.values() if v.success]
            typer.echo(f"[SUCCESS] Model '{model}' pushed successfully -> {handles}")
        else:
            typer.echo(f"[ERROR] Model push failed: {'; '.join(res.errors)}")
            raise typer.Exit(code=1)


@sync_app.command("pull")
def sync_pull(
    model: str = typer.Argument(..., help="Target model key to download from Kaggle Models Hub."),
    variation: str = typer.Option(
        "all",
        "--variation",
        "-v",
        help=(
            "Variation to pull ('all', 'pytorch/default', 'pytorch/fp32', 'onnx/fp16', "
            "'onnx/fp32')."
        ),
    ),
    force_download: bool = typer.Option(
        False, "--force", "-f", help="Force re-download even if already present locally."
    ),
    username: str | None = typer.Option(
        None, "--username", "-u", help="Optional Kaggle username override."
    ),
) -> None:
    """Download model variation(s) from Kaggle Models Hub into canonical directories."""
    sync_mgr = KaggleModelSyncManager(username=username)
    # Kaggle/transport/file errors, plus RuntimeError raised when kagglehub is not installed.
    pull_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), RuntimeError)
    vars_to_pull = list(VARIATION_BY_KEY.keys()) if variation == "all" else [variation]
    total = len(vars_to_pull)

    typer.echo(f"[START] Pulling {total} variation(s) for '{model}' from Kaggle Models Hub...")
    succeeded = 0
    failed = 0

    for idx, var_key in enumerate(vars_to_pull, 1):
        handle = sync_mgr.get_handle(model, var_key)
        target_dir = sync_mgr.get_variation_dir(model, var_key)
        typer.echo(f"[{idx}/{total}] Pulling variation [{var_key}] <- {handle}...")
        try:
            ok = sync_mgr.pull_variation(
                model_key=model, variation=var_key, force_download=force_download
            )
            if ok:
                succeeded += 1
                typer.echo(f"  [SUCCESS] Downloaded to {target_dir}")
            else:
                failed += 1
                typer.echo(f"  [ERROR] Pull returned false for {var_key}")
        except pull_errors as e:
            # Per-variation boundary: report every variation's outcome instead of aborting the pull
            # batch.
            failed += 1
            logger.exception("Kaggle pull failed for model '%s' variation '%s'", model, var_key)
            typer.echo(f"  [ERROR] Failed to pull {var_key}: {e}")

    if succeeded == total:
        typer.echo(
            f"[SUCCESS] Model '{model}' pulled successfully from Kaggle ({total} variations)."
        )
    else:
        typer.echo(f"[ERROR] Pull incomplete for '{model}': {succeeded}/{total} succeeded.")
        raise typer.Exit(code=1)
