"""Kaggle kernel session status names and the polling cadence of a watched kernel.

The names are the members of ``kagglesdk.kernels.types.kernels_enums.KernelWorkerStatus``
(the enum the Kaggle API returns from ``kernels_status``). A test pins them to the installed
SDK, so a renamed member fails there instead of leaving a job polling forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUEUED = "QUEUED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
ERROR = "ERROR"
CANCEL_REQUESTED = "CANCEL_REQUESTED"
CANCEL_ACKNOWLEDGED = "CANCEL_ACKNOWLEDGED"

CANCELLED_STATUSES: frozenset[str] = frozenset({CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED})
"""A kernel cancelled on kaggle.com (or through the API) reports one of these."""
TERMINAL_STATUSES: frozenset[str] = frozenset({COMPLETE, ERROR, *CANCELLED_STATUSES})
"""Statuses after which the session produces no further logs or output."""


def status_name(response: Any) -> str:
    """Upper-case status name of a ``kernels_status`` response (or of a bare enum/string).

    ``KernelWorkerStatus.CANCEL_ACKNOWLEDGED`` and ``"cancel_acknowledged"`` both give
    ``CANCEL_ACKNOWLEDGED``.
    """
    raw = getattr(response, "status", response)
    name = getattr(raw, "name", None)
    text = str(name if name is not None else raw)
    return text.rsplit(".", maxsplit=1)[-1].strip().upper()


def failure_message(response: Any) -> str:
    """The failure message of a ``kernels_status`` response, or an empty string."""
    return str(
        getattr(response, "failure_message", None)
        or getattr(response, "failureMessage", None)
        or ""
    )


@dataclass
class PollBackoff:
    """Status poll interval: starts at ``initial`` and grows while nothing changes.

    Each unchanged poll multiplies the interval by ``factor`` up to ``maximum``; a status
    change resets it to ``initial``. Kaggle sessions run for hours, so a 5-second poll that
    downloads the whole log is wasteful; the default starts at 30 s and settles at 5 min.
    """

    initial: float = 30.0
    factor: float = 1.5
    maximum: float = 300.0
    current: float = 0.0

    def __post_init__(self) -> None:
        self.initial = max(1.0, float(self.initial))
        self.maximum = max(self.initial, float(self.maximum))
        self.current = self.initial

    def next_interval(self, changed: bool) -> float:
        """Seconds to wait before the next poll, after a poll that did (not) see a change."""
        if changed:
            self.current = self.initial
        else:
            self.current = min(self.maximum, self.current * self.factor)
        return self.current
