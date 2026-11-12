"""Command-line arguments shared by the trainer and the forex curriculum orchestrator."""

from __future__ import annotations

import argparse


def add_clean_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the ``--clean`` / ``--fresh`` switch (``dest="clean"``)."""
    parser.add_argument(
        "--clean",
        "--fresh",
        dest="clean",
        action="store_true",
        help=help_text,
    )


def add_override_frozen_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--override-frozen`` switch (``dest="override_frozen"``)."""
    parser.add_argument(
        "--override-frozen",
        dest="override_frozen",
        action="store_true",
        help=(
            "Resume although a frozen run setting (architecture, optimizer type, dataset, "
            "metric definitions, seed, precision) changed; logged and recorded in "
            "settings_changes.jsonl. Without it such a resume is refused (or use --clean)."
        ),
    )


def add_timeframes_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the ``--timeframes`` option (minutes, one or more integers)."""
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=None,
        help=help_text,
    )
