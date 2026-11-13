"""Typer option types shared by several ``lemtrain`` commands (one flag, one help text)."""

from __future__ import annotations

from typing import Annotated

import typer

from training.cli_args import (
    ENV_MANAGER_REF_FLAG,
    ENV_MANAGER_REF_HELP,
    GIT_REF_FLAG,
    GIT_REF_HELP,
    LIMIT_HOURS_FLAG,
    LIMIT_HOURS_HELP,
    MARGIN_MINUTES_FLAG,
    MARGIN_MINUTES_HELP,
    OVERRIDE_FROZEN_FLAG,
)

OverrideFrozenOption = Annotated[
    bool,
    typer.Option(
        OVERRIDE_FROZEN_FLAG,
        help="Resume although a frozen run setting changed (logged and recorded).",
    ),
]
"""The trainer's ``--override-frozen`` switch (local training and Kaggle kernels)."""

GitRefOption = Annotated[str | None, typer.Option(GIT_REF_FLAG, help=GIT_REF_HELP)]
"""Ref of the kernel's suite clone (``lemtrain kaggle bundle|launch``)."""
EnvManagerRefOption = Annotated[
    str | None, typer.Option(ENV_MANAGER_REF_FLAG, help=ENV_MANAGER_REF_HELP)
]
"""Ref of the kernel's env-manager clone (default: the suite ref)."""
LimitHoursOption = Annotated[
    float | None, typer.Option(LIMIT_HOURS_FLAG, min=0.0, help=LIMIT_HOURS_HELP)
]
"""Per-kernel ``kaggle_session.limit_hours`` override."""
MarginMinutesOption = Annotated[
    float | None, typer.Option(MARGIN_MINUTES_FLAG, min=0.0, help=MARGIN_MINUTES_HELP)
]
"""Per-kernel ``kaggle_session.margin_minutes`` override."""
