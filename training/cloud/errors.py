"""Shared exception sets for Kaggle cloud operations.

Kept dependency-light (only ``requests``; ``kagglehub`` is imported lazily) so legacy shims can
import it without loading the Kaggle SDKs eagerly.
"""

from __future__ import annotations

import importlib
import logging

import requests

logger = logging.getLogger("lemtrain.cloud.errors")

# Errors the Kaggle REST client raises for expected remote failures: HTTP/transport errors
# (requests), local file / auth-file problems (OSError) and payload validation (ValueError, KeyError).
KAGGLE_API_ERRORS: tuple[type[Exception], ...] = (
    requests.RequestException,
    OSError,
    ValueError,
    KeyError,
)


def kagglehub_errors() -> tuple[type[Exception], ...]:
    """Return KAGGLE_API_ERRORS plus every exception class published by ``kagglehub.exceptions``.

    The class names differ between kagglehub releases, so they are discovered from the module
    instead of being listed by name. This is the single source for kagglehub failure types.
    """
    try:
        exc_mod = importlib.import_module("kagglehub.exceptions")
    except ImportError as exc:
        logger.debug("kagglehub.exceptions unavailable (%s); using base Kaggle error set", exc)
        return KAGGLE_API_ERRORS
    found = [
        attr
        for attr in vars(exc_mod).values()
        if isinstance(attr, type) and issubclass(attr, Exception) and attr not in KAGGLE_API_ERRORS
    ]
    return (*KAGGLE_API_ERRORS, *found)
