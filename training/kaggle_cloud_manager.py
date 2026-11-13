"""Kaggle cloud manager legacy shim delegating to training.cloud."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from training.cli_args import add_kaggle_launch_arguments, add_override_frozen_argument
from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kaggle_client import KaggleAuthError, authenticated_kaggle_api
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.kernel_bundle import BundleRequest, model_kernel_handle
from training.cloud.kernel_launch import KernelLaunch, launch_kernel
from training.cloud.kernel_status import status_name
from training.cloud.variations import ALL_VARIATIONS, VARIATION_BY_KEY
from training.ecosystem import kaggle_accelerators
from training.kaggle_stream import stream_kernel_logs
from training.notebooks.launch_options import DEFAULT_LAUNCH, LaunchOptions

logger = logging.getLogger("lemtrain.kaggle_cloud_manager")


def _authenticated_api(username: str | None, key: str | None) -> tuple[Any, str]:
    """The authenticated Kaggle API and its username (one credential resolver)."""
    user, token = resolve_kaggle_credentials(override_user=username, override_key=key)
    return authenticated_kaggle_api(user, token), user


def launch_kaggle_training(
    model_name: str,
    username: str | None = None,
    key: str | None = None,
    accelerator: str | None = None,
    options: LaunchOptions = DEFAULT_LAUNCH,
) -> KernelLaunch:
    """Push the model's generated kernel bundle (notebook plus metadata) to Kaggle.

    ``options`` are the notebook launch options (clone refs, ``--override-frozen``, session
    window); the defaults clone ``main`` and keep ``config.yaml``'s ``kaggle_session``.
    """
    api, user = _authenticated_api(username, key)
    request = BundleRequest(
        model_name, kernel_owner=user, accelerator_id=accelerator, launch=options
    )
    launch = launch_kernel(api, request)
    logger.info("Kernel %s pushed (version %s)", launch.handle, launch.version_number)
    return launch


def monitor_kaggle_training(model_name: str, username: str | None = None) -> None:
    """Stream status and logs of the model's kernel to the terminal."""
    try:
        api, user = _authenticated_api(username, None)
    except KaggleAuthError as exc:
        logger.error("Kaggle monitoring needs credentials: %s", exc)
        return
    stream_kernel_logs(api, model_kernel_handle(user, model_name))


def pull_kaggle_artifacts(
    model_name: str,
    destination_dir: str | None = None,
    username: str | None = None,
) -> bool:
    """Download latest checkpoints and metrics from Kaggle Models via unified sync manager.

    ``destination_dir`` overrides the hub root under which ``<model_name>/`` is written
    (default: the sibling ``LemGendaryModels`` directory).
    """
    sync_mgr = KaggleModelSyncManager(username=username, hub_root=destination_dir)
    logger.info("Pulling Kaggle model variations for %s ...", model_name)
    try:
        return sync_mgr.pull_model(model_key=model_name)
    except (RuntimeError, *KAGGLE_API_ERRORS) as exc:
        logger.error("Failed pulling model %s: %s", model_name, exc)
        return False


def push_kaggle_artifacts(model_name: str, username: str | None = None) -> bool:
    """Stage and upload multi-variation model to Kaggle Models Hub via unified sync pipeline."""
    sync_mgr = KaggleModelSyncManager(username=username)
    res = sync_mgr.push_model(model_key=model_name, variation=ALL_VARIATIONS)
    return res.success


def push_kaggle_model(
    model_name: str,
    variation: str = ALL_VARIATIONS,
    dry_run: bool = False,
    username: str | None = None,
    key: str | None = None,
) -> bool:
    """Stage and upload multi-variation model to Kaggle Models hub."""
    sync_mgr = KaggleModelSyncManager(username=username, key=key)
    res = sync_mgr.push_model(
        model_key=model_name,
        variation=variation,
        dry_run=dry_run,
    )
    logger.info("Kaggle model push result for %s: %s", model_name, res)
    return res.success


def _launch_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> LaunchOptions:
    """The launch options of the parsed arguments; invalid values end in a usage error."""
    try:
        return LaunchOptions.from_values(
            args.override_frozen,
            args.git_ref,
            args.env_manager_ref,
            args.limit_hours,
            args.margin_minutes,
        )
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        sys.stderr.write(f"{parser.prog}: error: {exc}\n")
        raise SystemExit(2) from exc


def main() -> None:
    """CLI entry point for Kaggle cloud management."""
    parser = argparse.ArgumentParser(description="LemGendary Headless Kaggle Cloud Engine")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[
            "launch",
            "status",
            "monitor",
            "monitor_interactive",
            "pull",
            "push",
            "push_model",
            "cancel",
            "setup_auth",
        ],
    )
    parser.add_argument("--model", type=str, default="nima_technical", help="Model manifold name")
    parser.add_argument(
        "--variation",
        type=str,
        default=ALL_VARIATIONS,
        choices=[ALL_VARIATIONS, *VARIATION_BY_KEY],
        help="Variation to push",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate staging without pushing to Kaggle"
    )
    parser.add_argument("--username", type=str, default=None, help="Kaggle Username override")
    parser.add_argument("--key", type=str, default=None, help="Kaggle API Key override")
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        choices=[acc.id for acc in kaggle_accelerators()],
        help="Kaggle accelerator id from the ecosystem file (default: its default)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Hub root for pulled artifacts (pull only; default: sibling LemGendaryModels)",
    )
    add_override_frozen_argument(parser)
    add_kaggle_launch_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.action == "setup_auth":
        u, _ = resolve_kaggle_credentials()
        logger.info("Authentication verified for Kaggle user: %s", u)
    elif args.action == "launch":
        launch_kaggle_training(
            args.model,
            username=args.username,
            key=args.key,
            accelerator=args.accelerator,
            options=_launch_options(parser, args),
        )
    elif args.action in ("monitor", "monitor_interactive"):
        monitor_kaggle_training(args.model, username=args.username)
    elif args.action == "pull":
        pull_kaggle_artifacts(args.model, destination_dir=args.output_dir, username=args.username)
    elif args.action == "push":
        push_kaggle_artifacts(args.model, username=args.username)
    elif args.action == "push_model":
        push_kaggle_model(
            args.model,
            variation=args.variation,
            dry_run=args.dry_run,
            username=args.username,
            key=args.key,
        )
    elif args.action == "status":
        try:
            api, user = _authenticated_api(args.username, args.key)
            handle = model_kernel_handle(user, args.model)
            logger.info("Status for %s: %s", handle, status_name(api.kernels_status(handle)))
        except (KaggleAuthError, *KAGGLE_API_ERRORS) as exc:
            logger.error("Status check failed: %s", exc)


if __name__ == "__main__":
    main()
