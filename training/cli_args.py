"""Command-line arguments shared by the trainer, the forex orchestrator and Kaggle launches."""

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


OVERRIDE_FROZEN_FLAG = "--override-frozen"
"""Trainer switch accepting a changed frozen run setting on resume (default off)."""


def add_override_frozen_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--override-frozen`` switch (``dest="override_frozen"``)."""
    parser.add_argument(
        OVERRIDE_FROZEN_FLAG,
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


GIT_REF_FLAG = "--git-ref"
"""Kaggle launch option: suite (and default env-manager) ref the notebook clones."""
ENV_MANAGER_REF_FLAG = "--env-manager-ref"
"""Kaggle launch option: env-manager ref when it differs from ``--git-ref``."""
LIMIT_HOURS_FLAG = "--limit-hours"
"""Kaggle launch option: session limit overriding ``kaggle_session.limit_hours``."""
MARGIN_MINUTES_FLAG = "--margin-minutes"
"""Kaggle launch option: stop margin overriding ``kaggle_session.margin_minutes``."""

GIT_REF_HELP = "Branch, tag or full commit SHA the kernel clones the suite at (default: main)."
ENV_MANAGER_REF_HELP = "Ref the kernel clones the env-manager at (default: the --git-ref value)."
LIMIT_HOURS_HELP = (
    "Session limit in hours for this kernel (default: kaggle_session of config.yaml)."
)
MARGIN_MINUTES_HELP = (
    "Minutes before the limit the trainer stops (default: kaggle_session of config.yaml)."
)


def add_kaggle_launch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the clone ref and session window options of a Kaggle launch (all default off)."""
    parser.add_argument(GIT_REF_FLAG, dest="git_ref", default=None, help=GIT_REF_HELP)
    parser.add_argument(
        ENV_MANAGER_REF_FLAG, dest="env_manager_ref", default=None, help=ENV_MANAGER_REF_HELP
    )
    parser.add_argument(
        LIMIT_HOURS_FLAG, dest="limit_hours", type=float, default=None, help=LIMIT_HOURS_HELP
    )
    parser.add_argument(
        MARGIN_MINUTES_FLAG,
        dest="margin_minutes",
        type=float,
        default=None,
        help=MARGIN_MINUTES_HELP,
    )
