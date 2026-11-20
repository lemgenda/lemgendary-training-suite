"""Exponential backoff retry utilities for remote cloud operations.

Provides a deterministic retry wrapper with exponential backoff and jitter,
strictly typed to avoid catching blind exceptions.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger("lemtrain.cloud.retry")

T = TypeVar("T")


def retry_with_backoff(
    action: Callable[[], T],
    *,
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...],
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Execute ``action``, retrying on ``retryable_exceptions`` with exponential backoff.

    Args:
        action: Zero-argument callable to execute.
        retries: Maximum number of retry attempts after the first attempt.
        initial_delay: Delay in seconds before the first retry attempt.
        backoff_factor: Multiplier applied to the delay after each retry.
        max_delay: Ceiling for retry delay in seconds.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        on_retry: Optional callback invoked before sleeping on retry.

    Returns:
        The result of ``action()`` upon success.

    Raises:
        The last encountered exception if all retry attempts are exhausted, or any
        exception not in ``retryable_exceptions`` immediately.
    """
    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(1, retries + 2):
        try:
            return action()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt > retries:
                logger.error(
                    "Exhausted %d retry attempts. Final error: %s",
                    retries,
                    exc,
                )
                raise
            jitter = random.uniform(0.0, 0.5 * delay)
            sleep_time = min(delay + jitter, max_delay)
            logger.warning(
                "Attempt %d/%d failed with %s: %s. Retrying in %.2fs...",
                attempt,
                retries + 1,
                exc.__class__.__name__,
                exc,
                sleep_time,
            )
            if on_retry is not None:
                on_retry(attempt, exc, sleep_time)
            time.sleep(sleep_time)
            delay = min(delay * backoff_factor, max_delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited unexpectedly without result or exception.")
