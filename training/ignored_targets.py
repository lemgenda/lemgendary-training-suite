"""The class target the cross-entropy losses and accuracy skip (single definition).

A classification sample whose label names no class (missing label file or metadata, an
unknown folder name) gets :data:`IGNORE_TARGET` instead of a real class id, so it adds
nothing to the loss or the accuracy instead of training as class 0. :class:`UnlabeledLog`
counts such samples per dataset and logs the running count.

A real class id must stay below :data:`IGNORE_TARGET`, or its samples would be skipped as
unlabeled: :func:`check_class_count` refuses a classifier declaring more classes at setup,
and :func:`check_class_id` refuses such an id where a dataset resolves a sample's class.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch

from training.registry.class_names import declared_class_count

logger = logging.getLogger("lemtrain.data.unlabeled")

IGNORE_TARGET = 255
"""``ignore_index`` of the cross-entropy criterion; above every class id of the registry."""


class IgnoreTargetCollision(ValueError):
    """A class id the cross-entropy would skip: it is not below :data:`IGNORE_TARGET`."""


def check_class_count(model_info: Mapping[str, Any] | None, model_key: str) -> None:
    """Refuse a classifier declaring more classes than ids below :data:`IGNORE_TARGET`.

    The class ids are ``0 .. num_classes - 1``, so at most ``IGNORE_TARGET`` classes fit.
    """
    count = declared_class_count(model_info)
    if count is not None and count > IGNORE_TARGET:
        raise IgnoreTargetCollision(
            f"Model '{model_key}' declares {count} classes, but class ids must stay below "
            f"{IGNORE_TARGET}, the target of unlabeled samples (at most {IGNORE_TARGET} "
            "classes)."
        )


def check_class_id(class_idx: int, source: str) -> int:
    """``class_idx`` when it is below :data:`IGNORE_TARGET`; else :class:`IgnoreTargetCollision`."""
    if class_idx >= IGNORE_TARGET:
        raise IgnoreTargetCollision(
            f"Class id {class_idx} of {source} is not below {IGNORE_TARGET}, the target of "
            "unlabeled samples; such a sample would be skipped as unlabeled."
        )
    return class_idx


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
