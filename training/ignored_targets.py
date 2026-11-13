"""The class target the cross-entropy losses and accuracy skip (single definition).

A classification sample whose label names no class (missing label file or metadata, an
unknown folder name) gets :data:`IGNORE_TARGET` instead of a real class id, so it adds
nothing to the loss or the accuracy instead of training as class 0. :class:`UnlabeledLog`
counts such samples per dataset and logs the running count.
"""

from __future__ import annotations

import logging
from collections import Counter

import torch

logger = logging.getLogger("lemtrain.data.unlabeled")

IGNORE_TARGET = 255
"""``ignore_index`` of the cross-entropy criterion; above every class id of the registry."""


def labelled(targets: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the targets that name a class."""
    return targets != IGNORE_TARGET


class UnlabeledLog:
    """Count of unlabeled classification samples per dataset, logged at 1, 2, 4, 8, ...

    Each data-loading process (DataLoader worker) keeps its own count.
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def count(self, dataset: str) -> int:
        """Unlabeled samples of ``dataset`` seen by this process."""
        return self.counts[dataset]

    def record(self, dataset: str, model_key: str, source: str) -> None:
        """Count one unlabeled sample of ``dataset`` (``source`` names it in debug logs)."""
        self.counts[dataset] += 1
        count = self.counts[dataset]
        logger.debug("Unlabeled classification sample %s in %s", source, dataset)
        if count & (count - 1) == 0:
            logger.warning(
                "Dataset '%s' (model '%s'): %d unlabeled classification samples so far; "
                "their target is ignored (%d), not class 0.",
                dataset,
                model_key,
                count,
                IGNORE_TARGET,
            )
