"""Typer option types shared by several ``lemtrain`` commands (one flag, one help text)."""

from __future__ import annotations

from typing import Annotated

import typer

from training.cli_args import OVERRIDE_FROZEN_FLAG

OverrideFrozenOption = Annotated[
    bool,
    typer.Option(
        OVERRIDE_FROZEN_FLAG,
        help="Resume although a frozen run setting changed (logged and recorded).",
    ),
]
"""The trainer's ``--override-frozen`` switch (local training and Kaggle kernels)."""
