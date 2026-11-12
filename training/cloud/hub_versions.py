"""Read-only lookup of the latest Kaggle Models Hub version of a model instance.

The kernel bundle pins ``model_sources`` to it (resume input) and the Kaggle watchers pull
only when it is newer than the version already merged locally.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from training.cloud.errors import KAGGLE_API_ERRORS

logger = logging.getLogger("lemtrain.cloud.hub_versions")

VersionLookup = Callable[[str], "int | None"]
"""Latest version of ``<owner>/<model>/<framework>/<instance>``; None when unknown."""


def kaggle_version_lookup(api: Any) -> VersionLookup:
    """:data:`VersionLookup` backed by ``KaggleApi.model_instance_get`` (read-only)."""

    def latest_version(handle: str) -> int | None:
        try:
            instance = api.model_instance_get(handle)
        except KAGGLE_API_ERRORS as exc:
            logger.info("No Kaggle Hub version for %s: %s", handle, exc)
            return None
        version = getattr(instance, "version_number", None)
        return version if isinstance(version, int) and version > 0 else None

    return latest_version
