"""lemtrain sync: cloud storage probe, sync run, model push and pull."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import typer
import yaml

from training.cli.option_groups import option_group
from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.variations import ALL_VARIATIONS, VARIATION_BY_KEY, selected_variations
from training.registry import get_model_registry
from training.services.sync_service import CloudSyncService

logger = logging.getLogger("lemtrain.cli.lemtrain")


VARIATION_CHOICES: list[str] = [ALL_VARIATIONS, *VARIATION_BY_KEY]
"""Accepted ``--variation`` values (the variation keys from ``training.cloud.variations``)."""


def checked_variation(value: str) -> str:
    """``--variation`` callback: accept only :data:`VARIATION_CHOICES` (a usage error else)."""
    if value not in VARIATION_CHOICES:
        raise typer.BadParameter(f"'{value}' is not one of {VARIATION_CHOICES}")
    return value


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


@dataclass(frozen=True)
class PushOptions:
    """Options of ``lemtrain sync push``."""

    all_models: Annotated[
        bool,
        typer.Option("--all", "-a", help="Push all 22 registered models to Kaggle Models Hub."),
    ] = False
    variation: Annotated[
        str,
        typer.Option(
            "--variation",
            "-v",
            callback=checked_variation,
            help=f"Variation to push: {', '.join(VARIATION_CHOICES)}.",
        ),
    ] = ALL_VARIATIONS
    version_notes: Annotated[
        str | None, typer.Option("--notes", "-n", help="Optional release version notes.")
    ] = None
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate isolation and metadata without uploading to Kaggle."
        ),
    ] = False
    source_dir: Annotated[
        str | None,
        typer.Option(
            "--source-dir",
            "-s",
            help=(
                "Optional LemGendaryModels root to stage and push from (default: sibling of repo)."
            ),
        ),
    ] = None
    username: Annotated[
        str | None, typer.Option("--username", "-u", help="Optional Kaggle username override.")
    ] = None

    def push_kwargs(self) -> dict:
        """The ``KaggleModelSyncManager.push_model`` options."""
        return {
            "variation": self.variation,
            "version_notes": self.version_notes,
            "dry_run": self.dry_run,
        }


@sync_app.command("push")
@option_group("opts", PushOptions)
def sync_push(
    model: str | None = typer.Argument(
        None, help="Model key to upload to Kaggle Models (optional if --all is specified)."
    ),
    *,
    opts: PushOptions,
) -> None:
    """Stage and upload local model checkpoints, variations, and metrics to Kaggle Models."""
    if not model and not opts.all_models:
        typer.echo("[ERROR] Please specify either a model key or --all.")
        raise typer.Exit(code=1)

    sync_mgr = KaggleModelSyncManager(username=opts.username, hub_root=opts.source_dir)
    push_opts = opts.push_kwargs()

    if opts.all_models:
        _push_fleet(sync_mgr, push_opts)
        return
    if not model:
        typer.echo(
            "[ERROR] No target model specified. Provide a --model argument or use --all-models.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"[START] Staging and pushing '{model}' to Kaggle Models (variation: {opts.variation}, "
        f"dry_run: {opts.dry_run})..."
    )
    res = _stage_and_push(sync_mgr, model, push_opts)
    if res.success:
        handles = [v.handle for v in res.variations.values() if v.success]
        typer.echo(f"[SUCCESS] Model '{model}' pushed successfully -> {handles}")
    else:
        typer.echo(f"[ERROR] Model push failed: {'; '.join(res.errors)}")
        raise typer.Exit(code=1)


def _stage_and_push(sync_mgr: KaggleModelSyncManager, model_key: str, push_opts: dict):
    """Run the one push policy (stage, model page, uploads, metadata) for ``model_key``."""
    res = sync_mgr.push_model(model_key=model_key, **push_opts)
    unchanged = [key for key, v in res.variations.items() if v.skipped]
    if unchanged:
        typer.echo(f"  * Unchanged since the last upload (no new version): {unchanged}")
    return res


def _push_fleet(sync_mgr: KaggleModelSyncManager, push_opts: dict) -> None:
    """Push every registered model; exit 1 only when a real run pushed none."""
    # Kaggle/transport/file errors, metadata YAML errors, and RuntimeError (kagglehub missing).
    push_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), yaml.YAMLError, RuntimeError)
    models = get_model_registry().list_models()
    total_models = len(models)
    typer.echo(
        f"[START] Commencing fleet Kaggle push for all {total_models} models (variation: "
        f"{push_opts['variation']}, dry_run: {push_opts['dry_run']})..."
    )
    succeeded = 0
    failed = 0

    for idx, m_spec in enumerate(models, 1):
        m_key = m_spec.key
        typer.echo(f"[{idx}/{total_models}] Processing '{m_key}' ({m_spec.name})...")
        try:
            res = _stage_and_push(sync_mgr, m_key, push_opts)
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
    if failed > 0 and not push_opts["dry_run"] and succeeded == 0:
        raise typer.Exit(code=1)


def _pull_variation(
    sync_mgr: KaggleModelSyncManager, model: str, var_key: str, force_download: bool
) -> bool:
    """Pull one variation and report its outcome; True on success."""
    target_dir = sync_mgr.pull_destination(model, var_key)
    # Kaggle/transport/file errors, plus RuntimeError raised when kagglehub is not installed.
    pull_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), RuntimeError)
    try:
        ok = sync_mgr.pull_variation(
            model_key=model, variation=var_key, force_download=force_download
        )
    except pull_errors as e:
        # Per-variation boundary: report every variation's outcome instead of aborting the pull
        # batch.
        logger.exception("Kaggle pull failed for model '%s' variation '%s'", model, var_key)
        typer.echo(f"  [ERROR] Failed to pull {var_key}: {e}")
        return False
    if ok:
        typer.echo(f"  [SUCCESS] Downloaded to {target_dir}")
    else:
        typer.echo(f"  [ERROR] Pull returned false for {var_key}")
    return bool(ok)


@sync_app.command("pull")
def sync_pull(
    model: str = typer.Argument(..., help="Target model key to download from Kaggle Models Hub."),
    variation: str = typer.Option(
        ALL_VARIATIONS,
        "--variation",
        "-v",
        callback=checked_variation,
        help=f"Variation to pull: {', '.join(VARIATION_CHOICES)}.",
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
    vars_to_pull = selected_variations(variation)
    total = len(vars_to_pull)

    typer.echo(f"[START] Pulling {total} variation(s) for '{model}' from Kaggle Models Hub...")
    succeeded = 0
    failed = 0

    for idx, var_key in enumerate(vars_to_pull, 1):
        typer.echo(
            f"[{idx}/{total}] Pulling variation [{var_key}] <- "
            f"{sync_mgr.get_handle(model, var_key)}..."
        )
        if _pull_variation(sync_mgr, model, var_key, force_download):
            succeeded += 1
        else:
            failed += 1

    if succeeded == total:
        typer.echo(
            f"[SUCCESS] Model '{model}' pulled successfully from Kaggle ({total} variations)."
        )
    else:
        typer.echo(f"[ERROR] Pull incomplete for '{model}': {succeeded}/{total} succeeded.")
        raise typer.Exit(code=1)
