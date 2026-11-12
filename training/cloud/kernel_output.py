"""Retrieve a finished kernel's model payload and merge it into the local hub folder.

The training notebook keeps ``LemGendaryModels/<model>`` under the platform working dir,
which Kaggle saves as kernel output. Only the ``pytorch/default`` payload (checkpoints,
``README.md``, ``metrics.csv``) and the session deadline marker are downloaded; the merge
rules are :func:`training.cloud.model_pull.merge_downloaded_payload` (newer stored epoch
wins, nothing local is deleted). The marker is only read, never merged.
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
from training.notebooks.platform_paths import MODELS_FOLDER

logger = logging.getLogger("lemtrain.cloud.kernel_output")


@dataclass(frozen=True)
class KernelOutput:
    """What the retrieval merged, and whether the session stopped at its deadline."""

    report: MergeReport
    deadline_stop: bool


def output_pattern(model_key: str) -> str:
    """``kernels_output`` file pattern of the model's payload files and deadline marker."""
    folder = re.escape(f"{MODELS_FOLDER}/{model_key}/")
    marker = re.escape(DEADLINE_MARKER)
    return rf"(^|/){folder}(checkpoints/[^/]+\.pth|README\.md|metrics\.csv|{marker})$"


def retrieve_kernel_output(
    api: Any, kernel_handle: str, model_key: str, hub_root: Path
) -> KernelOutput:
    """Download the model payload of ``kernel_handle``'s output and merge it into the hub."""
    hub_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".kernel-output-{model_key}-", dir=hub_root) as tmp:
        api.kernels_output(kernel_handle, path=tmp, file_pattern=output_pattern(model_key))
        found = sorted(Path(tmp).rglob(f"{MODELS_FOLDER}/{model_key}"))
        if not found:
            logger.info("Kernel %s produced no payload for %s", kernel_handle, model_key)
            return KernelOutput(MergeReport(), deadline_stop=False)
        deadline_stop = (found[0] / DEADLINE_MARKER).is_file()
        report = merge_downloaded_payload(
            found[0], hub_root / model_key, force=False, keep=is_default_payload_file
        )
    logger.info(
        "Kernel %s output merged into %s (deadline stop: %s)",
        kernel_handle,
        hub_root / model_key,
        deadline_stop,
    )
    return KernelOutput(report, deadline_stop)
