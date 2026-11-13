"""Retrieve a finished kernel's model payload and merge it into the local hub folder.

The training notebook keeps ``LemGendaryModels/<model>`` under the platform working dir,
which Kaggle saves as kernel output. Only the ``pytorch/default`` payload (checkpoints,
``README.md``, ``metrics.csv``), the session deadline marker and the resume refusal record
(:mod:`training.run_settings.refusal`) are downloaded; the merge rules are
:func:`training.cloud.model_pull.merge_downloaded_payload` (newer stored epoch wins, nothing
local is deleted). The marker and the record are only read, never merged.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.cloud.model_pull import MergeReport, merge_downloaded_payload
from training.cloud.session_deadline import DEADLINE_MARKER
from training.cloud.variations import is_default_payload_file
from training.platform_paths import MODELS_FOLDER
from training.run_settings.refusal import RESUME_REFUSAL_RECORD, ResumeRefusalRecord

logger = logging.getLogger("lemtrain.cloud.kernel_output")


@dataclass(frozen=True)
class KernelOutput:
    """What the retrieval merged, whether the session stopped at its deadline, and the
    trainer's resume refusal text (``None`` when it did not refuse to resume)."""

    report: MergeReport
    deadline_stop: bool
    resume_refusal: str | None = None


def _model_files_pattern(model_key: str, names: str) -> str:
    """``kernels_output`` file pattern of ``names`` (a regex) in the model's folder."""
    folder = re.escape(f"{MODELS_FOLDER}/{model_key}/")
    return rf"(^|/){folder}({names})$"


def output_pattern(model_key: str) -> str:
    """``kernels_output`` pattern of the payload files, deadline marker and refusal record."""
    marker = re.escape(DEADLINE_MARKER)
    refusal = re.escape(RESUME_REFUSAL_RECORD)
    return _model_files_pattern(
        model_key, rf"checkpoints/[^/]+\.pth|README\.md|metrics\.csv|{marker}|{refusal}"
    )


def _model_folder(downloaded: Path, model_key: str) -> Path | None:
    """The model folder inside a ``kernels_output`` download, if the kernel wrote one."""
    found = sorted(downloaded.rglob(f"{MODELS_FOLDER}/{model_key}"))
    return found[0] if found else None


def retrieve_kernel_output(
    api: Any, kernel_handle: str, model_key: str, hub_root: Path
) -> KernelOutput:
    """Download the model payload of ``kernel_handle``'s output and merge it into the hub."""
    hub_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".kernel-output-{model_key}-", dir=hub_root) as tmp:
        api.kernels_output(kernel_handle, path=tmp, file_pattern=output_pattern(model_key))
        model_dir = _model_folder(Path(tmp), model_key)
        if model_dir is None:
            logger.info("Kernel %s produced no payload for %s", kernel_handle, model_key)
            return KernelOutput(MergeReport(), deadline_stop=False)
        deadline_stop = (model_dir / DEADLINE_MARKER).is_file()
        refusal = ResumeRefusalRecord.in_folder(model_dir).read()
        report = merge_downloaded_payload(
            model_dir, hub_root / model_key, force=False, keep=is_default_payload_file
        )
    logger.info(
        "Kernel %s output merged into %s (deadline stop: %s)",
        kernel_handle,
        hub_root / model_key,
        deadline_stop,
    )
    return KernelOutput(report, deadline_stop, refusal)


def retrieve_resume_refusal(api: Any, kernel_handle: str, model_key: str) -> str | None:
    """Download only the resume refusal record of ``kernel_handle``'s output; its text."""
    pattern = _model_files_pattern(model_key, re.escape(RESUME_REFUSAL_RECORD))
    with tempfile.TemporaryDirectory(prefix=f".kernel-refusal-{model_key}-") as tmp:
        api.kernels_output(kernel_handle, path=tmp, file_pattern=pattern)
        model_dir = _model_folder(Path(tmp), model_key)
        if model_dir is None:
            return None
        return ResumeRefusalRecord.in_folder(model_dir).read()
