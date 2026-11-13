"""lemtrain kaggle: accelerators, kernel bundles, kernel launch and status."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer

from training.cli.option_groups import option_group
from training.cli.shared_options import (
    EnvManagerRefOption,
    GitRefOption,
    LimitHoursOption,
    MarginMinutesOption,
    OverrideFrozenOption,
)
from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kaggle_client import KaggleAuthError, authenticated_kaggle_api
from training.cloud.kernel_bundle import (
    BundleRequest,
    KernelBundleError,
    build_kernel_bundle,
    model_kernel_handle,
)
from training.cloud.kernel_launch import KernelLaunchError, launch_kernel
from training.cloud.kernel_status import status_name
from training.ecosystem import kaggle_accelerator, kaggle_accelerators
from training.notebooks.launch_options import LaunchOptions

kaggle_app = typer.Typer(
    name="kaggle", help="Kaggle kernel bundles (generated notebook plus kernel-metadata.json)."
)

_ACCELERATOR_HELP = "Accelerator id from 'lemtrain kaggle accelerators' (default: the default)."


@kaggle_app.command("accelerators")
def kaggle_list_accelerators() -> None:
    """List the Kaggle accelerators of the ecosystem file."""
    default_id = kaggle_accelerator().id
    for acc in kaggle_accelerators():
        marker = " (default)" if acc.id == default_id else ""
        typer.echo(f"{acc.id:<12} {acc.label:<14} {acc.machine_shape or 'CPU'}{marker}")


@dataclass(frozen=True)
class LaunchFlags:
    """Notebook launch options of ``lemtrain kaggle bundle|launch`` (all keep the defaults)."""

    override_frozen: OverrideFrozenOption = False
    git_ref: GitRefOption = None
    env_manager_ref: EnvManagerRefOption = None
    limit_hours: LimitHoursOption = None
    margin_minutes: MarginMinutesOption = None

    def options(self) -> LaunchOptions:
        """The validated launch options (raises ``ValueError``)."""
        return LaunchOptions.from_values(**asdict(self))


@kaggle_app.command("bundle")
@option_group("flags", LaunchFlags)
def kaggle_bundle(
    model: Annotated[str, typer.Argument(help="Model key.")],
    *,
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder to write the bundle into.")],
    accelerator: Annotated[
        str | None, typer.Option("--accelerator", "-a", help=_ACCELERATOR_HELP)
    ] = None,
    resume_version: Annotated[
        int | None,
        typer.Option(
            "--resume-version", help="pytorch/default Hub version to mount (model_sources)."
        ),
    ] = None,
    flags: LaunchFlags,
) -> None:
    """Write the kernel bundle of MODEL without pushing it (for review or a manual push)."""
    user, _ = resolve_kaggle_credentials()
    try:
        request = BundleRequest(model, user, accelerator, resume_version, flags.options())
        bundle = build_kernel_bundle(request, out)
    except (KernelBundleError, ValueError) as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(bundle.metadata, indent=2))
    typer.echo(f"[OK] Kernel bundle written to {bundle.directory}")


@kaggle_app.command("launch")
@option_group("flags", LaunchFlags)
def kaggle_launch(
    model: str = typer.Argument(..., help="Model key."),
    *,
    accelerator: str | None = typer.Option(None, "--accelerator", "-a", help=_ACCELERATOR_HELP),
    flags: LaunchFlags,
) -> None:
    """Push the kernel bundle of MODEL to Kaggle (runs the session)."""
    user, token = resolve_kaggle_credentials()
    try:
        request = BundleRequest(model, user, accelerator, launch=flags.options())
        api = authenticated_kaggle_api(user, token)
        launch = launch_kernel(api, request)
    except (
        ValueError,
        KaggleAuthError,
        KernelBundleError,
        KernelLaunchError,
        *KAGGLE_API_ERRORS,
    ) as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"[OK] Kernel {launch.handle} pushed (version {launch.version_number}) {launch.url}")


@kaggle_app.command("status")
def kaggle_status(model: str = typer.Argument(..., help="Model key.")) -> None:
    """Print the session status of MODEL's kernel."""
    user, token = resolve_kaggle_credentials()
    try:
        api = authenticated_kaggle_api(user, token)
        handle = model_kernel_handle(user, model)
        typer.echo(f"{handle}: {status_name(api.kernels_status(handle))}")
    except (KaggleAuthError, KernelBundleError, *KAGGLE_API_ERRORS) as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1) from exc
