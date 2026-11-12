"""Progress hooks of a Kaggle model push (GUI job telemetry, CLI output).

:meth:`training.cloud.kaggle_model_sync.KaggleModelSyncManager.push_model` is the one push
policy (stage, model page first, skip unchanged, upload with license and notes, metadata).
Callers that report progress pass an observer instead of re-implementing the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from training.cloud.kaggle_model_sync import VariationPushResult


class PushObserver(Protocol):
    """Receives the steps of one model push."""

    def staged(self, model_key: str, files: list[str]) -> None:
        """Variations were staged into ``dist/kaggle`` (``files``: staged file names)."""
        ...

    def variation_started(self, index: int, total: int, variation: str, handle: str) -> None:
        """Variation ``index`` of ``total`` is about to upload; may raise to cancel."""
        ...

    def variation_done(self, index: int, total: int, result: VariationPushResult) -> None:
        """Variation ``index`` of ``total`` finished (uploaded, skipped or failed)."""
        ...
