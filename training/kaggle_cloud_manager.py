"""Kaggle cloud manager legacy shim delegating to training.cloud."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.manager import CloudSyncError

logger = logging.getLogger("lemtrain.kaggle_cloud_manager")


def get_kernel_slug(model_name: str, username: str) -> str:
    """Generate standardized Kaggle kernel slug."""
    mgr = KaggleHubManager(username=username)
    return mgr.get_kernel_slug(model_name)


def create_cloud_kernel_bundle(model_name: str, username: str, gpu: str = "T4") -> Path:
    """Package a standalone Kaggle kernel bundle containing metadata and execution script."""
    mgr = KaggleHubManager(username=username)
    return mgr.create_cloud_kernel_bundle(model_name, gpu=gpu)


def launch_kaggle_training(
    model_name: str,
    config: dict[str, Any] | None = None,
    username: str | None = None,
    key: str | None = None,
    gpu: str = "T4",
) -> bool:
    """Pushes and launches a training kernel to Kaggle GPU Cloud headlessly."""
    user, _ = resolve_kaggle_credentials(override_user=username, override_key=key)
    mgr = KaggleHubManager(username=user, key=key, config=config)
    kernel_dir = mgr.create_cloud_kernel_bundle(model_name, gpu=gpu)
    slug = mgr.get_kernel_slug(model_name)

    logger.info("Deploying kernel bundle %s to Kaggle with GPU %s...", slug, gpu)

    try:
        kaggle_mod = importlib.import_module("kaggle.api.kaggle_api_extended")
        api_cls = getattr(kaggle_mod, "KaggleApi")
        api = api_cls()
        api.authenticate()
        api.kernels_push(str(kernel_dir))
        logger.info("Kernel successfully pushed to Kaggle!")
        return True
    except (ImportError, Exception) as err:
        logger.warning("Kaggle Python SDK push notice: %s. Trying CLI fallback...", err)

    try:
        res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            logger.info("CLI push successful: %s", res.stdout.strip())
            return True
        logger.error("CLI push failed: %s", res.stderr.strip())
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.error("CLI fallback failed: %s", exc)

    return False


def monitor_kaggle_training(model_name: str, username: str | None = None, poll_interval: int = 5) -> None:
    """Stream status and logs from Kaggle for specified model kernel."""
    user, _ = resolve_kaggle_credentials(override_user=username)
    mgr = KaggleHubManager(username=user)
    slug = mgr.get_kernel_slug(model_name)

    logger.info("Connecting to telemetry stream for %s...", slug)
    try:
        km = importlib.import_module("training.kaggle_monitor")
        kaggle_mod = importlib.import_module("kaggle.api.kaggle_api_extended")
        api_cls = getattr(kaggle_mod, "KaggleApi")
        api = api_cls()
        api.authenticate()
        km.stream_kernel_logs(api, slug, poll_interval=poll_interval)
    except Exception as exc:
        logger.error("Kaggle monitoring stream error: %s", exc)


def pull_kaggle_artifacts(
    model_name: str,
    destination_dir: str | None = None,
    username: str | None = None,
) -> bool:
    """Download latest checkpoints and metrics from Kaggle Models."""
    user, _ = resolve_kaggle_credentials(override_user=username)
    mgr = KaggleHubManager(username=user)
    suite_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(destination_dir) if destination_dir else suite_dir.parent / "LemGendaryModels" / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    handle = mgr.get_handle(model_name)
    logger.info("Pulling Kaggle artifacts for %s (%s) -> %s", model_name, handle, target_dir)

    try:
        return mgr.pull(handle, target_dir)
    except CloudSyncError as exc:
        logger.warning("Model registry pull notice: %s. Attempting kernel output fallback...", exc)

    try:
        kaggle_mod = importlib.import_module("kaggle.api.kaggle_api_extended")
        api_cls = getattr(kaggle_mod, "KaggleApi")
        api = api_cls()
        api.authenticate()
        slug = mgr.get_kernel_slug(model_name)
        api.kernels_output(slug, path=str(target_dir))
        logger.info("Kernel output artifacts pulled successfully.")
        return True
    except Exception as k_err:
        logger.error("Artifact pull failed: %s", k_err)
        return False


def main() -> None:
    """CLI entry point for Kaggle cloud management."""
    parser = argparse.ArgumentParser(description="LemGendary Headless Kaggle Cloud Engine")
    parser.add_argument("--action", type=str, required=True, choices=["launch", "status", "monitor", "monitor_interactive", "pull", "cancel", "setup_auth"])
    parser.add_argument("--model", type=str, default="nima_technical", help="Model manifold name")
    parser.add_argument("--username", type=str, default=None, help="Kaggle Username override")
    parser.add_argument("--key", type=str, default=None, help="Kaggle API Key override")
    parser.add_argument("--gpu", type=str, default="T4", choices=["P100", "T4", "T4x2"], help="Kaggle GPU accelerator")
    parser.add_argument("--output_dir", type=str, default=None, help="Target destination for downloaded artifacts")
    args = parser.parse_args()

    if args.action == "setup_auth":
        u, _ = resolve_kaggle_credentials()
        logger.info("Authentication verified for Kaggle user: %s", u)
    elif args.action == "launch":
        launch_kaggle_training(args.model, username=args.username, key=args.key, gpu=args.gpu)
    elif args.action in ("monitor", "monitor_interactive"):
        monitor_kaggle_training(args.model, username=args.username)
    elif args.action == "pull":
        pull_kaggle_artifacts(args.model, destination_dir=args.output_dir, username=args.username)
    elif args.action == "status":
        u, _ = resolve_kaggle_credentials(override_user=args.username)
        slug = get_kernel_slug(args.model, u)
        try:
            kaggle_mod = importlib.import_module("kaggle.api.kaggle_api_extended")
            api_cls = getattr(kaggle_mod, "KaggleApi")
            api = api_cls()
            api.authenticate()
            st = api.kernels_status(slug)
            logger.info("Status for %s: %s", slug, st)
        except Exception as exc:
            logger.error("Status check failed: %s", exc)


if __name__ == "__main__":
    main()
