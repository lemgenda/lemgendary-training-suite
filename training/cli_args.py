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


def add_timeframes_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the ``--timeframes`` option (minutes, one or more integers)."""
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=None,
        help=help_text,
    )
