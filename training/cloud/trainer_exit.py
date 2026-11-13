"""Trainer exit gate of the generated training notebooks (Kaggle and Colab).

The training cell runs the trainer as a subprocess. Its exit code decides whether the later
cells (release publish, Google Drive mirror) may run:

* exit :data:`TRAINER_OK_EXIT_CODE` without a session deadline marker: training completed;
* exit :data:`TRAINER_OK_EXIT_CODE` with :data:`~training.cloud.session_deadline.
  DEADLINE_MARKER` in the model folder: an intentional stop at the session deadline that the
  next session resumes. It is not a failure; the publish cell skips itself on the marker. The
  deadline stop keeps exit code 0 so ``tools/launch_ddp.py`` (torchrun) and the forex
  curriculum, which only pass 0 through as success, report it unchanged;
* any other exit code: the trainer failed, was interrupted or refused to resume.
  :func:`check_trainer_exit` raises :class:`TrainerExitError`, which stops the notebook, so
  the kernel ends with an error and nothing is published. A resume refusal keeps its
  ``resume_refusal.json`` record (written by the trainer, read by the Kaggle job watcher)
  and its text is the exception message.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from training.cloud.session_deadline import DEADLINE_MARKER
from training.run_settings.refusal import ResumeRefusalRecord

logger = logging.getLogger("lemtrain.cloud.trainer_exit")

TRAINER_OK_EXIT_CODE = 0
"""Exit code of a completed run and of a run stopped at its session deadline."""


class TrainerOutcome(Enum):
    """How a trainer subprocess that exited with :data:`TRAINER_OK_EXIT_CODE` ended."""

    COMPLETED = "completed"
    DEADLINE_STOP = "deadline_stop"


class TrainerExitError(RuntimeError):
    """The trainer exited non-zero; the notebook must not run its publish cells."""

    def __init__(self, returncode: int, refusal: str | None) -> None:
        detail = refusal if refusal is not None else "see the training log above"
        super().__init__(f"Trainer exited with code {returncode}: {detail}")
        self.returncode = returncode
        self.refusal = refusal


def check_trainer_exit(returncode: int, model_dir: Path) -> TrainerOutcome:
    """Classify the trainer exit; raise :class:`TrainerExitError` on a non-zero exit.

    Args:
        returncode: The trainer subprocess exit code (negative for a signal).
        model_dir: The model's ``LemGendaryModels/<model>`` folder of the session.

    Raises:
        TrainerExitError: ``returncode`` is not :data:`TRAINER_OK_EXIT_CODE`; carries the
            resume refusal text when the trainer recorded one.
    """
    if returncode != TRAINER_OK_EXIT_CODE:
        refusal = ResumeRefusalRecord.in_folder(model_dir).read()
        logger.error("Trainer exited with code %d (resume refusal: %s)", returncode, refusal)
        raise TrainerExitError(returncode, refusal)
    if (model_dir / DEADLINE_MARKER).is_file():
        logger.info("Trainer stopped at the session deadline; the next session resumes it")
        return TrainerOutcome.DEADLINE_STOP
    return TrainerOutcome.COMPLETED
