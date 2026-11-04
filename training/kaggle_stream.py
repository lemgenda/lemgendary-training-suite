"""Kaggle kernel log parsing and live status/log streaming with optional checkpoint pulls."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from training.cloud.errors import KAGGLE_API_ERRORS
from training.kaggle_cloud_manager import pull_kaggle_artifacts

logger = logging.getLogger("lemtrain.cloud.kaggle_monitor")

# pull_kaggle_artifacts reports most failures itself; RuntimeError covers a missing kagglehub SDK.
_PULL_ERRORS: tuple[type[Exception], ...] = (RuntimeError, *KAGGLE_API_ERRORS)

# Kernel states after which the run produces no further logs.
TERMINAL_STATES = ("COMPLETE", "ERROR", "CANCELACK")

# A kernel log line that marks a finished epoch or a saved checkpoint.
_EPOCH_SIGNAL = re.compile(
    r"epoch\s+\d+\s+complete|saving.*checkpoint|model version.*committed|\[sota\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AutoPull:
    """Pull checkpoints of ``model_name`` into LemGendaryModels as the kernel saves them."""

    model_name: str
    username: str | None = None

    def pull(self, notice: str) -> None:
        """Pull the model artifacts; report a failed pull as ``[WARN] <notice> notice``."""
        try:
            pull_kaggle_artifacts(self.model_name, username=self.username)
        except _PULL_ERRORS as pull_err:
            print(f"[WARN] {notice} notice: {pull_err}")


def is_epoch_milestone(line: str) -> bool:
    """Return True when a kernel log line reports a finished epoch or a saved checkpoint."""
    return _EPOCH_SIGNAL.search(line) is not None


def fetch_new_log_lines(api: Any, kernel_slug: str, printed_count: int) -> tuple[list[str], int]:
    """Return the kernel log lines after ``printed_count`` and the new printed count."""
    all_lines = parse_kernel_logs(api.kernels_logs(kernel_slug)).splitlines()
    return all_lines[printed_count:], max(printed_count, len(all_lines))


def _join_stream_items(items: list[Any], keep_plain: bool) -> str:
    """Join the ``data`` of stream objects; plain items are kept only when ``keep_plain``."""
    chunks = []
    for item in items:
        if isinstance(item, dict) and "data" in item:
            chunks.append(str(item["data"]))
        elif keep_plain:
            chunks.append(str(item))
    return "".join(chunks)


def parse_kernel_logs(raw_logs: Any) -> str:
    """
    Parses raw API log response (JSON array of stream objects or plain string) into text lines.
    """
    if not raw_logs:
        return ""

    if isinstance(raw_logs, str):
        try:
            entries = json.loads(raw_logs)
            if isinstance(entries, list):
                return _join_stream_items(entries, keep_plain=False)
            if isinstance(entries, dict) and "data" in entries:
                return str(entries["data"])
        except (ValueError, TypeError):
            return raw_logs
    elif isinstance(raw_logs, list):
        return _join_stream_items(raw_logs, keep_plain=True)

    return str(raw_logs)


def _status_name(st: Any) -> str:
    """Return the upper-case kernel status name of a ``kernels_status`` response."""
    return str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()


def _failure_message(st: Any) -> Any:
    """Return the failure message of a ``kernels_status`` response, if any."""
    return getattr(st, "failure_message", None) or getattr(st, "failureMessage", None)


def _print_banner(kernel_slug: str, pull: AutoPull | None) -> None:
    """Print the monitoring header and the auto-pull mode."""
    print("\n" + "=" * 76)
    print(f"  [TELEMETRY] MONITORING KERNEL: {kernel_slug}")
    if pull:
        print(f"  [AUTO-PULL ACTIVE] Checkpoints will sync to LemGendaryModels/{pull.model_name}")
    else:
        print("  [MONITOR ONLY] Checkpoint auto-pulling is disabled.")
    print("=" * 76)


def _print_persisted_logs(api: Any, kernel_slug: str) -> None:
    """Print the stored logs of a finished kernel (the last 100 lines when longer)."""
    try:
        log_text = parse_kernel_logs(api.kernels_logs(kernel_slug))
        if log_text.strip():
            lines = log_text.splitlines()
            print(f"--- Persisted Execution Logs ({len(lines)} lines) ---")
            if len(lines) > 100:
                print(f"[NOTE] Displaying trailing 100 lines of {len(lines)} total lines:")
                lines = lines[-100:]
            for line in lines:
                print(line)
            print("--- End of Logs ---")
        else:
            print("[INFO] No log output available for this kernel run.")
    except KAGGLE_API_ERRORS as log_exc:
        print(f"[WARN] Failed to fetch execution logs: {log_exc}")


def _report_finished_kernel(api: Any, kernel_slug: str, st: Any, pull: AutoPull | None) -> None:
    """Print the logs and failure details of a kernel that already finished, then pull."""
    current_status = _status_name(st)
    print(f"[INFO] Kernel is in terminal state: {current_status}")
    _print_persisted_logs(api, kernel_slug)

    failure_msg = _failure_message(st)
    if failure_msg:
        print(f"\n[FAILURE DETAILS] {failure_msg}")

    if pull and current_status == "COMPLETE":
        print(f"\n[SYNC] Performing artifact pull for completed run >> {pull.model_name} <<...")
        pull.pull("Artifact pull")


def _print_new_lines(api: Any, kernel_slug: str, printed_count: int, pull: AutoPull | None) -> int:
    """Print the new log lines, pulling on epoch milestones; return the printed count."""
    try:
        new_lines, printed_count = fetch_new_log_lines(api, kernel_slug, printed_count)
        for line in new_lines:
            print(line)
            # Check for epoch completion or checkpoint save if auto-pull is enabled
            if pull and is_epoch_milestone(line):
                print(
                    f"\n[{time.strftime('%H:%M:%S')}] [AUTO-PULL] New "
                    f"checkpoint detected. Syncing >> {pull.model_name} <<..."
                )
                pull.pull("Auto-pull")
    except KAGGLE_API_ERRORS as exc:
        logger.debug("Log poll failed for %s; retrying next interval: %s", kernel_slug, exc)
    return printed_count


def _report_terminal_status(st: Any, status: str, pull: AutoPull | None) -> None:
    """Print the final status of a live-streamed kernel and run the final pull."""
    print(f"\n[{time.strftime('%H:%M:%S')}] [TERMINAL] Cloud job finished with status: {status}")
    failure_msg = _failure_message(st)
    if failure_msg:
        print(f"[FAILURE DETAILS] {failure_msg}")

    if pull and status == "COMPLETE":
        print(
            "\n[SYNC] Training complete. Performing final artifact pull for >> "
            f"{pull.model_name} <<..."
        )
        pull.pull("Final artifact pull")


def _poll_until_terminal(
    api: Any, kernel_slug: str, poll_interval: int, pull: AutoPull | None, st: Any
) -> None:
    """Poll status and logs every ``poll_interval`` seconds until the kernel finishes."""
    last_status = _status_name(st)
    printed_count = 0
    while True:
        try:
            st = api.kernels_status(kernel_slug)
            status = _status_name(st)
            if status != last_status:
                print(f"\n[{time.strftime('%H:%M:%S')}] [SIGNAL] Cloud Status: {status}")
                last_status = status
        except KAGGLE_API_ERRORS as exc:
            logger.debug(
                "Status poll failed for %s; keeping last status %s: %s",
                kernel_slug,
                last_status,
                exc,
            )
            status = last_status

        printed_count = _print_new_lines(api, kernel_slug, printed_count, pull)

        if status in TERMINAL_STATES:
            _report_terminal_status(st, status, pull)
            break

        time.sleep(poll_interval)


def stream_kernel_logs(
    api: Any, kernel_slug: str, poll_interval: int = 5, pull: AutoPull | None = None
) -> None:
    """
    Streams live kernel logs and status telemetry to terminal.
    With ``pull``, pulls checkpoints and weights to local LemGendaryModels per completed
    epoch; without it, operates in monitor-only mode with zero checkpoint pulling.
    """
    _print_banner(kernel_slug, pull)

    try:
        st = api.kernels_status(kernel_slug)
        current_status = _status_name(st)
    except KAGGLE_API_ERRORS as exc:
        print(f"[ERROR] Could not fetch status for {kernel_slug}: {exc}")
        return

    print(f"Current Status: {current_status}")
    print("Press Ctrl+C at any time to return to menu.\n")

    if current_status in TERMINAL_STATES:
        _report_finished_kernel(api, kernel_slug, st, pull)
        return

    # Kernel is running or queued: live polling loop
    print(f"[LIVE] Streaming telemetry every {poll_interval}s...")
    try:
        _poll_until_terminal(api, kernel_slug, poll_interval, pull, st)
    except KeyboardInterrupt:
        print(
            f"\n\n[INFO] Detached from telemetry stream for {kernel_slug}. Process continues "
            "on Kaggle."
        )
