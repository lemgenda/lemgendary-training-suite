"""lemtrain sync: cloud storage probe, sync run, model push and pull."""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from training.cli.option_groups import option_group
from training.cloud.errors import kagglehub_errors
from training.cloud.kaggle_model_sync import (
    KaggleModelSyncManager,
    ModelPushResult,
    VariationPullResult,
)
from training.cloud.sync_report import ModelSyncReport, VariationSyncReport
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


def push_all_help() -> str:
    """``--all`` help naming the registry's current model count (never a fixed number)."""
    return (
        f"Push all {len(get_model_registry().list_keys())} registered models to Kaggle Models Hub."
    )


@dataclass(frozen=True)
class PushOptions:
    """Options of ``lemtrain sync push``."""

    all_models: Annotated[
        bool,
        typer.Option("--all", "-a", help=push_all_help()),
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
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Bypass idempotent no-op check and force upload to Kaggle."
        ),
    ] = False
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Validate isolation and metadata without uploading to Kaggle."
        ),
    ] = False
    report: Annotated[
        str | None,
        typer.Option("--report", help="Path to write machine-readable JSON sync report."),
    ] = None
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON sync report to stdout."),
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
            "force": self.force,
        }


@dataclass(frozen=True)
class PullOptions:
    """Options of ``lemtrain sync pull``."""

    variation: Annotated[
        str,
        typer.Option(
            "--variation",
            "-v",
            callback=checked_variation,
            help=f"Variation to pull: {', '.join(VARIATION_CHOICES)}.",
        ),
    ] = ALL_VARIATIONS
    force_download: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Force re-download even if already present locally."
        ),
    ] = False
    version: Annotated[
        str | None,
        typer.Option(
            "--version", help="Pin specific model variation version (e.g. 2, '2', '@v2')."
        ),
    ] = None
    report: Annotated[
        str | None,
        typer.Option("--report", help="Path to write machine-readable JSON sync report."),
    ] = None
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON sync report to stdout."),
    ] = False
    username: Annotated[
        str | None, typer.Option("--username", "-u", help="Optional Kaggle username override.")
    ] = None


def _parse_version(version_str: str | None) -> int | None:
    """Parse version option string ('2', 'v2', '@v2') into integer or None."""
    if not version_str:
        return None
    cleaned = version_str.strip().lstrip("@").lstrip("v")
    if not cleaned.isdigit():
        raise typer.BadParameter(f"Version must be a positive integer, got '{version_str}'")
    return int(cleaned)


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

    start_time = time.time()
    sync_mgr = KaggleModelSyncManager(username=opts.username, hub_root=opts.source_dir)
    push_opts = opts.push_kwargs()

    if opts.all_models:
        _push_fleet(sync_mgr, push_opts, report_path=opts.report, json_out=opts.json_output)
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
    elapsed = time.time() - start_time

    # Build machine-readable report
    rep = ModelSyncReport(
        action="push",
        model_key=model,
        success=res.success,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 3),
        errors=list(getattr(res, "errors", [])),
        metadata=res.to_dict().get("metadata_sync") or {}
        if hasattr(res, "to_dict")
        else {},
    )
    variations_dict = getattr(res, "variations", {}) or {}
    for k, v in variations_dict.items():
        rep.variations[k] = VariationSyncReport(
            variation=getattr(v, "variation", k),
            handle=getattr(v, "handle", ""),
            success=getattr(v, "success", False),
            status="noop"
            if getattr(v, "skipped", False)
            else ("success" if getattr(v, "success", False) else "failed"),
            files=getattr(v, "files", []),
            checksums=getattr(v, "checksums", {}),
            checksum_verified=bool(getattr(v, "checksums", {})),
            idempotent_noop=getattr(v, "skipped", False),
            retries=getattr(v, "retries", 0),
            error=getattr(v, "error", None),
        )

    if opts.report:
        rep.write_to(opts.report)
        typer.echo(f"[REPORT] Wrote push report to {opts.report}")
    if opts.json_output:
        typer.echo(rep.to_json())

    if res.success:
        handles = [v.handle for v in res.variations.values() if v.success]
        typer.echo(f"[SUCCESS] Model '{model}' pushed successfully -> {handles}")
    else:
        typer.echo(f"[ERROR] Model push failed: {'; '.join(res.errors)}")
        raise typer.Exit(code=1)


def _stage_and_push(
    sync_mgr: KaggleModelSyncManager, model_key: str, push_opts: dict
) -> ModelPushResult:
    """Run the one push policy (stage, model page, uploads, metadata) for ``model_key``."""
    res = sync_mgr.push_model(model_key=model_key, **push_opts)
    unchanged = [key for key, v in res.variations.items() if v.skipped]
    if unchanged:
        typer.echo(f"  * Unchanged since the last upload (no new version): {unchanged}")
    return res


def _emit_fleet_report(
    fleet_rep: dict[str, Any], report_path: str | None, json_out: bool
) -> None:
    if report_path:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fleet_rep, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(f"[REPORT] Wrote fleet push report to {report_path}")
    if json_out:
        typer.echo(json.dumps(fleet_rep, indent=2, sort_keys=True))


def _push_fleet(
    sync_mgr: KaggleModelSyncManager,
    push_opts: dict,
    report_path: str | None = None,
    json_out: bool = False,
) -> None:
    """Push every registered model; exit 1 only when a real run pushed none."""
    start_time = time.time()
    models = get_model_registry().list_models()
    typer.echo(
        f"[START] Commencing fleet Kaggle push for all {len(models)} models (variation: "
        f"{push_opts['variation']}, dry_run: {push_opts['dry_run']})..."
    )
    succeeded, failed, all_reports = 0, 0, {}
    for idx, m_spec in enumerate(models, 1):
        typer.echo(f"[{idx}/{len(models)}] Processing '{m_spec.key}' ({m_spec.name})...")
        try:
            res = _stage_and_push(sync_mgr, m_spec.key, push_opts)
            all_reports[m_spec.key] = res.to_dict()
            if res.success:
                succeeded += 1
                handles = [v.handle for v in res.variations.values() if v.success]
                typer.echo(f"  [SUCCESS] Pushed -> {handles}")
            else:
                failed += 1
                typer.echo(f"  [WARN] Failed/skipped: {'; '.join(res.errors)}")
        except (*kagglehub_errors(), yaml.YAMLError, RuntimeError) as e:
            failed += 1
            logger.exception("Fleet Kaggle push failed for model '%s'", m_spec.key)
            typer.echo(f"  [ERROR] Exception for '{m_spec.key}': {e}")
            all_reports[m_spec.key] = {"model_key": m_spec.key, "success": False, "error": str(e)}

    fleet_rep = {
        "action": "fleet_push",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "total_models": len(models),
        "succeeded": succeeded,
        "failed": failed,
        "models": all_reports,
    }
    _emit_fleet_report(fleet_rep, report_path, json_out)
    typer.echo(
        f"[COMPLETED] Fleet push finished: {succeeded} succeeded, {failed} failed/skipped out "
        f"of {len(models)} models."
    )
    if failed > 0 and not push_opts["dry_run"] and succeeded == 0:
        raise typer.Exit(code=1)


def _pull_variation(
    sync_mgr: KaggleModelSyncManager,
    model: str,
    var_key: str,
    force_download: bool,
    version: int | None = None,
) -> VariationPullResult:
    """Pull one variation and report its outcome as a VariationPullResult."""
    pull_errors: tuple[type[Exception], ...] = (*kagglehub_errors(), RuntimeError)
    try:
        return sync_mgr.pull_variation_detailed(
            model_key=model,
            variation=var_key,
            force_download=force_download,
            version=version,
        )
    except pull_errors as e:
        logger.exception("Kaggle pull failed for model '%s' variation '%s'", model, var_key)
        target_dir = sync_mgr.pull_destination(model, var_key)
        return VariationPullResult(
            variation=var_key,
            handle=sync_mgr.get_handle(model, var_key),
            success=False,
            destination=target_dir,
            error=str(e),
            version=version,
        )


def _pull_selected_variations(
    sync_mgr: KaggleModelSyncManager,
    model: str,
    vars_to_pull: list[str],
    force_download: bool,
    pinned_version: int | None,
) -> tuple[int, dict[str, VariationPullResult]]:
    succeeded = 0
    var_results: dict[str, VariationPullResult] = {}
    for idx, var_key in enumerate(vars_to_pull, 1):
        handle = sync_mgr.get_handle(model, var_key)
        if pinned_version is not None:
            handle = f"{handle}/{pinned_version}"
        typer.echo(f"[{idx}/{len(vars_to_pull)}] Pulling variation [{var_key}] <- {handle}...")
        res = _pull_variation(
            sync_mgr, model, var_key, force_download, version=pinned_version
        )
        var_results[var_key] = res
        if res.success:
            succeeded += 1
            typer.echo(
                f"  [SUCCESS] Downloaded to {res.destination} "
                f"(verified checksums: {res.checksum_verified})"
            )
        else:
            typer.echo(f"  [ERROR] Failed to pull {var_key}: {res.error}")
    return succeeded, var_results


@sync_app.command("pull")
@option_group("opts", PullOptions)
def sync_pull(
    model: str = typer.Argument(..., help="Target model key to download from Kaggle Models Hub."),
    *,
    opts: PullOptions,
) -> None:
    """Download model variation(s) from Kaggle Models Hub into canonical directories."""
    start_time = time.time()
    pinned_version = _parse_version(opts.version)
    sync_mgr = KaggleModelSyncManager(username=opts.username)
    vars_to_pull = selected_variations(opts.variation)
    ver_suffix = f" (version {pinned_version})" if pinned_version is not None else ""
    typer.echo(
        f"[START] Pulling {len(vars_to_pull)} variation(s) for '{model}'{ver_suffix} "
        "from Kaggle Models Hub..."
    )
    succeeded, var_results = _pull_selected_variations(
        sync_mgr, model, vars_to_pull, opts.force_download, pinned_version
    )
    elapsed = time.time() - start_time
    is_success = succeeded == len(vars_to_pull)

    # Build machine-readable pull report
    rep = ModelSyncReport(
        action="pull",
        model_key=model,
        success=is_success,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 3),
        errors=[f"[{k}] {v.error}" for k, v in var_results.items() if not v.success],
    )
    for k, v in var_results.items():
        rep.variations[k] = VariationSyncReport(
            variation=v.variation,
            handle=v.handle,
            success=v.success,
            status="success" if v.success else "failed",
            files=v.files_written,
            checksums=v.checksums,
            checksum_verified=v.checksum_verified,
            retries=v.retries,
            version=v.version,
            error=v.error,
        )

    if opts.report:
        rep.write_to(opts.report)
        typer.echo(f"[REPORT] Wrote pull report to {opts.report}")
    if opts.json_output:
        typer.echo(rep.to_json())

    if is_success:
        typer.echo(
            f"[SUCCESS] Model '{model}' pulled successfully from Kaggle "
            f"({len(vars_to_pull)} variations)."
        )
    else:
        typer.echo(
            f"[ERROR] Pull incomplete for '{model}': {succeeded}/{len(vars_to_pull)} succeeded."
        )
        raise typer.Exit(code=1)
