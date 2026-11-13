"""Terminal monitor of a Kaggle kernel: status, logs and optional Hub-version pulls.

The loop is :class:`training.cloud.kernel_watch.KernelWatcher` (the GUI job uses the same
one); this module only prints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from training.cloud.errors import KAGGLE_API_ERRORS, kagglehub_errors
from training.cloud.hub_versions import kaggle_version_lookup
from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.kernel_status import (
    COMPLETE,
    TERMINAL_STATUSES,
    PollBackoff,
    failure_message,
    status_name,
)
from training.cloud.kernel_watch import KernelWatcher, WatchHooks, parse_kernel_logs

logger = logging.getLogger("lemtrain.cloud.kaggle_monitor")


@dataclass(frozen=True)
class AutoPull:
    """Pull ``pytorch/default`` of ``model_name`` whenever the Hub has a newer version."""

    model_name: str
    username: str | None = None

    def pull(self, api: Any, notice: str) -> None:
        """Pull a newer Hub version, if any; report a failed pull as ``[WARN] <notice>``."""
        try:
            version = KaggleModelSyncManager(username=self.username).pull_default_if_newer(
                self.model_name, kaggle_version_lookup(api)
            )
        except (RuntimeError, *kagglehub_errors()) as pull_err:
            print(f"[WARN] {notice} notice: {pull_err}")
            return
        if version is not None:
            print(f"[AUTO-PULL] Hub version {version} of {self.model_name} merged locally.")


def _print_banner(kernel_slug: str, pull: AutoPull | None) -> None:
    """Print the monitoring header and the auto-pull mode."""
    print("\n" + "=" * 76)
    print(f"  [TELEMETRY] MONITORING KERNEL: {kernel_slug}")
    if pull:
        print(f"  [AUTO-PULL ACTIVE] Newer Hub versions sync to LemGendaryModels/{pull.model_name}")
    else:
        print("  [MONITOR ONLY] Checkpoint auto-pulling is disabled.")
    print("=" * 76)


def _print_persisted_logs(api: Any, kernel_slug: str) -> None:
    """Print the stored logs of a finished kernel (the last 100 lines when longer)."""
    try:
        lines = parse_kernel_logs(api.kernels_logs(kernel_slug)).splitlines()
    except KAGGLE_API_ERRORS as log_exc:
        print(f"[WARN] Failed to fetch execution logs: {log_exc}")
        return
    if not lines:
        print("[INFO] No log output available for this kernel run.")
        return
    print(f"--- Persisted Execution Logs ({len(lines)} lines) ---")
    if len(lines) > 100:
        print(f"[NOTE] Displaying trailing 100 lines of {len(lines)} total lines:")
        lines = lines[-100:]
    for line in lines:
        print(line)
    print("--- End of Logs ---")


def _report_end(api: Any, status: str, failure: str, pull: AutoPull | None) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] [TERMINAL] Kernel status: {status}")
    if failure:
        print(f"[FAILURE DETAILS] {failure}")
    if pull and status == COMPLETE:
        print(f"\n[SYNC] Pulling the final Hub version of >> {pull.model_name} <<...")
        pull.pull(api, "Final artifact pull")


def stream_kernel_logs(
    api: Any, kernel_slug: str, poll_interval: int = 30, pull: AutoPull | None = None
) -> None:
    """Stream status and logs of ``kernel_slug`` to the terminal until it ends.

    With ``pull``, ``pytorch/default`` is pulled whenever the Hub has a newer version;
    without it the monitor never downloads anything.
    """
    _print_banner(kernel_slug, pull)
    try:
        response = api.kernels_status(kernel_slug)
    except KAGGLE_API_ERRORS as exc:
        print(f"[ERROR] Could not fetch status for {kernel_slug}: {exc}")
        return
    current = status_name(response)
    print(f"Current Status: {current}")
    if current in TERMINAL_STATUSES:
        _print_persisted_logs(api, kernel_slug)
        _report_end(api, current, failure_message(response), pull)
        return

    print(f"[LIVE] Polling every {poll_interval}s (backing off while nothing changes)...")
    print("Press Ctrl+C at any time to return to menu.\n")
    hooks = WatchHooks(emit=print, on_poll=(lambda: pull.pull(api, "Auto-pull")) if pull else None)
    watcher = KernelWatcher(api, kernel_slug, PollBackoff(initial=poll_interval), hooks)
    try:
        result = watcher.watch(current)
    except KeyboardInterrupt:
        print(f"\n\n[INFO] Detached from {kernel_slug}. The kernel continues on Kaggle.")
        return
    _report_end(api, result.status or "", result.failure, pull)
