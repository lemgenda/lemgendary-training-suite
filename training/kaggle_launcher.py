"""Interactive Kaggle menus: launch training kernels and pick kernels to monitor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.kernel_bundle import BundleRequest, KernelBundleError
from training.cloud.kernel_launch import KernelLaunchError, launch_kernel
from training.ecosystem import kaggle_accelerator
from training.kaggle_accounts import (
    fetch_user_kernels,
    get_registered_models,
    match_model_from_slug,
)
from training.kaggle_stream import AutoPull, stream_kernel_logs

# Legacy or non-specific machine shapes that are replaced by the default accelerator's shape.
_GENERIC_SHAPES = ["gpu", "none", "cpu"]


def _default_machine_shape() -> str:
    """Machine shape of the ecosystem default accelerator (``""`` for CPU)."""
    return kaggle_accelerator().machine_shape or ""


_GPU_ABORT = (
    "    print('[ERROR] [CRITICAL] NO GPU DETECTED! Training aborted to preserve "
    "quota.')\n    sys.exit(1)"
)
_GPU_WARN = (
    "    print('[WARNING] NO GPU DETECTED!')\n    "
    "print('   -> Continuing in fallback execution mode...')"
)


def _print_kernel_list(kernels: list[dict[str, Any]], empty_msg: str) -> None:
    """Print the numbered kernel list with active kernels tagged."""
    if not kernels:
        print(empty_msg)
        return
    for idx, k in enumerate(kernels, start=1):
        status_tag = f"[{k['status']}]"
        if k["status"] in ["RUNNING", "QUEUED"]:
            status_tag = f"[ACTIVE: {k['status']}]"
        slug_name = k["ref"].split("/")[-1]
        print(f"  {idx}. {slug_name:<40} {status_tag}")


def _pick_monitor_slug(choice: str, kernels: list[dict[str, Any]], username: str) -> str | None:
    """Resolve a monitor-menu choice to a kernel slug (a listed or a typed one)."""
    if not choice.isdigit():
        return None
    val = int(choice)
    if 1 <= val <= len(kernels):
        return kernels[val - 1]["ref"]
    if val != len(kernels) + 1:
        return None
    raw_input = input("\nEnter kernel slug (e.g. username/kernel-name or kernel-name): ").strip()
    if not raw_input:
        return None
    return raw_input if "/" in raw_input else f"{username}/{raw_input}"


def interactive_kernel_monitor(api: Any, username: str) -> None:
    """
    Submenu for selecting which kernel to monitor (Monitor Only: no pulling).
    """
    while True:
        print("\n" + "=" * 76)
        print(f"  [MONITOR ONLY] ACTIVE & RECENT KERNELS FOR: {username}")
        print("=" * 76)
        print("Querying Kaggle for active and recent kernels...")

        kernels = fetch_user_kernels(api, username, limit=20)
        _print_kernel_list(kernels, "  No recent kernels found on Kaggle for this user.")

        custom_opt_idx = len(kernels) + 1
        print(f"\n  {custom_opt_idx}. [CUSTOM] Enter custom kernel slug / model name")
        print("  R. [REFRESH] Re-query kernel statuses")
        print("  B. [BACK] Return to User Selection")
        print()

        choice = input(f"Select kernel to monitor (1-{custom_opt_idx}, R, B): ").strip()

        if choice.upper() in ["B", "Q"]:
            break
        if choice.upper() == "R":
            continue

        target_slug = _pick_monitor_slug(choice, kernels, username)
        if target_slug:
            # Monitor Only mode: no checkpoint pulls
            stream_kernel_logs(api, target_slug)
            input("\nPress Enter to return to kernels list...")


def _prepare_kernel_metadata(kernel_dir: Path) -> dict[str, Any]:
    """Force GPU, internet and privacy on the pulled kernel metadata; return it."""
    meta_file = kernel_dir / "kernel-metadata.json"
    meta_data: dict[str, Any] = {}
    if not meta_file.exists():
        return meta_data
    try:
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        meta_data["enable_gpu"] = "true"
        meta_data["enable_internet"] = "true"
        if "is_private" not in meta_data:
            meta_data["is_private"] = "true"
        raw_shape = str(meta_data.get("machine_shape", "")).strip()
        # Normalize legacy/non-specific shapes ('Gpu', empty) to an explicit accelerator
        if not raw_shape or raw_shape.lower() in _GENERIC_SHAPES:
            meta_data["machine_shape"] = _default_machine_shape()
        meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError) as meta_err:
        # ValueError: JSONDecodeError; TypeError: non-object JSON.
        print(f"[REMEDY] Metadata update notice: {meta_err}")
    return meta_data


def _soften_gpu_abort(nb_json: dict[str, Any]) -> bool:
    """Replace the hard no-GPU abort in code cells with a warning; True when changed."""
    modified_nb = False
    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "NO GPU DETECTED! Training aborted to preserve quota." in src and "sys.exit(1)" in src:
            # Replace hard abort with the non-fatal warning used by notebook_generator
            cell["source"] = src.replace(_GPU_ABORT, _GPU_WARN)
            modified_nb = True
    return modified_nb


def _patch_notebook_sentinels(kernel_dir: Path) -> None:
    """Harden notebook sentinel cells against a premature ``sys.exit(1)`` on start-up."""
    try:
        for nb_path in kernel_dir.glob("*.ipynb"):
            nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
            if _soften_gpu_abort(nb_json):
                nb_path.write_text(json.dumps(nb_json, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError, AttributeError) as nb_patch_err:
        # ValueError: JSONDecodeError; Type/AttributeError: malformed notebook cells.
        print(f"[REMEDY] Notebook sentinel sync notice: {nb_patch_err}")


def _relaunch_kernel(api: Any, selected_slug: str) -> bool:
    """Pull a notebook, harden it and push it back to run on a Kaggle GPU."""
    print(f"\n[LAUNCH] Pulling notebook definition for '{selected_slug}'...")
    with tempfile.TemporaryDirectory() as td:
        try:
            api.kernels_pull(selected_slug, path=td, metadata=True)
            meta_data = _prepare_kernel_metadata(Path(td))
            _patch_notebook_sentinels(Path(td))

            chosen_acc = str(meta_data.get("machine_shape") or _default_machine_shape())
            if chosen_acc.lower() in _GENERIC_SHAPES:
                chosen_acc = _default_machine_shape()
            print(
                "[LAUNCH] Pushing kernel bundle to trigger execution on Kaggle "
                f"GPU ({chosen_acc})..."
            )
            api.kernels_push(td, acc=chosen_acc)
            print(f"[SUCCESS] Kernel '{selected_slug}' pushed and queued on Kaggle GPU!")
        except KAGGLE_API_ERRORS as push_err:
            print(f"[ERROR] Failed to launch kernel: {push_err}")
            input("\nPress Enter to continue...")
            return False
    return True


def _launch_existing(api: Any, selected_slug: str) -> tuple[str | None, str | None]:
    """Relaunch a listed notebook; return its slug and the model to pull checkpoints for."""
    target_model = match_model_from_slug(selected_slug)
    if not target_model:
        prompt_m = input(
            "Enter target model key for checkpoint saving (e.g. mirnet_exposure): "
        ).strip()
        if prompt_m:
            target_model = prompt_m
    if not _relaunch_kernel(api, selected_slug):
        return None, None
    return selected_slug, target_model


def _deploy_new_model(api: Any, username: str) -> tuple[str | None, str | None]:
    """Bundle a registered local model as a kernel and push it; return slug and model."""
    models = get_registered_models()
    print("\n--- SELECT LOCAL MODEL MANIFOLD TO DEPLOY ---")
    for m_idx, m_name in enumerate(models, start=1):
        print(f"  {m_idx}. {m_name}")
    m_choice = input(f"Select model (1-{len(models)}): ").strip()
    if not (m_choice.isdigit() and 1 <= int(m_choice) <= len(models)):
        return None, None
    target_model = models[int(m_choice) - 1]
    return _launch_generated(api, username, target_model)


def _launch_generated(api: Any, username: str, model_key: str) -> tuple[str | None, str | None]:
    """Push the model's generated kernel bundle; return its handle and the model."""
    print(f"[LAUNCH] Building and pushing the kernel bundle of '{model_key}'...")
    try:
        launch = launch_kernel(api, BundleRequest(model_key, kernel_owner=username))
    except (KernelBundleError, KernelLaunchError, *KAGGLE_API_ERRORS) as push_err:
        print(f"[ERROR] Failed to push kernel: {push_err}")
        input("\nPress Enter to continue...")
        return None, None
    print(f"[SUCCESS] Kernel '{launch.handle}' pushed (version {launch.version_number}).")
    return launch.handle, model_key


def interactive_train_launcher(api: Any, username: str) -> None:
    """
    Submenu for launching training on Kaggle and streaming logs with per-epoch checkpoint pulling.
    """
    while True:
        print("\n" + "=" * 76)
        print(f"  [TRAIN ON KAGGLE] SELECT NOTEBOOK FOR USER: {username}")
        print("=" * 76)
        print("Querying user's Kaggle notebooks...")

        kernels = fetch_user_kernels(api, username, limit=20)
        _print_kernel_list(kernels, "  No existing notebooks found on Kaggle.")

        new_opt_idx = len(kernels) + 1
        print(
            f"\n  {new_opt_idx}. [DEPLOY NEW MODEL] Deploy and launch a registered model from "
            "local suite"
        )
        print("  B. [BACK] Return to User Selection")
        print()

        choice = input(f"Select notebook to run (1-{new_opt_idx}, B): ").strip()

        if choice.upper() in ["B", "Q"]:
            break

        selected_slug: str | None = None
        target_model: str | None = None
        if choice.isdigit() and 1 <= int(choice) <= len(kernels):
            selected_slug, target_model = _launch_existing(api, kernels[int(choice) - 1]["ref"])
        elif choice.isdigit() and int(choice) == new_opt_idx:
            selected_slug, target_model = _deploy_new_model(api, username)

        if selected_slug:
            # Train mode: pull checkpoints of the target model as epochs finish
            pull = AutoPull(target_model, username) if target_model else None
            stream_kernel_logs(api, selected_slug, pull=pull)
            input("\nPress Enter to return to notebooks list...")
