"""Exponential backoff retry utilities for remote cloud operations.

Provides a deterministic retry wrapper with exponential backoff and jitter,
strictly typed to avoid catching blind exceptions.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger("lemtrain.cloud.retry")

T = TypeVar("T")


@dataclass(frozen=True)
class BackoffPolicy:
    """Exponential backoff configuration for :func:`retry_with_backoff`."""

    retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 30.0


def retry_with_backoff(
    action: Callable[[], T],
    *,
    retryable_exceptions: tuple[type[Exception], ...],
    policy: BackoffPolicy = BackoffPolicy(),
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute ``action``, retrying on ``retryable_exceptions`` with exponential backoff."""
    delay = policy.initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, policy.retries + 2):
        try:
            return action()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt > policy.retries:
                logger.error(
                    "Exhausted %d retry attempts. Final error: %s",
                    policy.retries,
                    exc,
                )
                raise
            jitter = random.uniform(0.0, 0.5 * delay)
            sleep_time = min(delay + jitter, policy.max_delay)
            logger.warning(
                "Attempt %d/%d failed with %s: %s. Retrying in %.2fs...",
                attempt,
                policy.retries + 1,
                exc.__class__.__name__,
                exc,
                sleep_time,
            )
            if on_retry is not None:
                on_retry(attempt, exc, sleep_time)
            time.sleep(sleep_time)
            delay = min(delay * policy.backoff_factor, policy.max_delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited unexpectedly without result or exception.")
