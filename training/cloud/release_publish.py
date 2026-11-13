"""Release publish of a trained model: export every format, then the one push policy.

The notebook's publish cell calls :func:`publish_release` instead of running its own upload
loop, so a notebook release and a GUI or CLI push create Hub versions the same way
(:meth:`training.cloud.kaggle_model_sync.KaggleModelSyncManager.push_model`: model page first,
license, unchanged variations skipped, version notes with epoch, metrics and suite version).
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.cloud.kaggle_model_sync import KaggleModelSyncManager, ModelPushResult
from training.export import ExportError, export_all

logger = logging.getLogger("lemtrain.cloud.release_publish")

RELEASE_EXPORT_ERRORS: tuple[type[Exception], ...] = (
    ExportError,
    OSError,
    RuntimeError,
    ValueError,
)
"""Export failures that stop a release (nothing is uploaded after a failed export)."""


def publish_release(
    model_key: str,
    hub_root: Path | str | None = None,
    *,
    dry_run: bool = False,
    manager: KaggleModelSyncManager | None = None,
) -> ModelPushResult:
    """Export ``model_key`` into its hub folder, then push all four variations.

    ``hub_root`` defaults to the one hub root (on Kaggle the kernel output). An export failure
    returns a failed result and uploads nothing, so a release never mixes new checkpoints
    with stale exports.
    """
    publisher = manager or KaggleModelSyncManager(hub_root=hub_root)
    model_dir = publisher.get_model_dir(model_key)
    try:
        exported = export_all(model_key, output_dir=model_dir)
    except RELEASE_EXPORT_ERRORS as exc:
        message = f"Export failed; nothing was published: {exc}"
        logger.error("%s (model=%s)", message, model_key)
        return ModelPushResult(model_key=model_key, success=False, errors=[message])
    logger.info("Exported %s: %s", model_key, sorted(exported))
    return publisher.push_model(model_key, dry_run=dry_run)
