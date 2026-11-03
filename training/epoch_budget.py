"""Epoch budget policy for the training engine.

Decides, after every completed epoch, whether the engine continues, extends
its epoch budget, or stops. The policy is pure (no torch, no I/O) so the
loop-control contract can be unit-tested in isolation.

Stop conditions, checked in this order:

1. ``converged``: the 4-pillar completion invariant is satisfied.
2. ``early_stop``: the governor signalled early stopping for this epoch.
3. ``budget_reached``: the configured epoch budget is spent and no extension
   is allowed (extensions disabled, or the governor did not report that the
   model is still improving).
4. ``extension_limit``: the budget is spent and the configured extension
   ceiling (``training_budget.max_extension_epochs``) is exhausted.

Extensions are opt-in. They are configured in ``config.yaml`` under
``training_budget`` and are disabled when that section is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

CONFIG_SECTION = "training_budget"


class EpochAction(str, Enum):
    """Loop-control action returned by the policy."""

    CONTINUE = "continue"
    EXTEND = "extend"
    STOP = "stop"


class StopReason(str, Enum):
    """Why the training loop ended."""

    CONVERGED = "converged"
    EARLY_STOP = "early_stop"
    BUDGET_REACHED = "budget_reached"
    EXTENSION_LIMIT = "extension_limit"


@dataclass(frozen=True)
class EpochBudgetConfig:
    """Budget-extension settings from the ``training_budget`` config section.

    Attributes:
        extension_epochs: Epochs added per extension.
        max_extension_epochs: Total extra epochs allowed across all
            extensions. ``0`` makes the configured epoch budget a hard cap.
    """

    extension_epochs: int = 0
    max_extension_epochs: int = 0

    @property
    def extensions_enabled(self) -> bool:
        """Whether the budget may be extended at all."""
        return self.extension_epochs > 0 and self.max_extension_epochs > 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> EpochBudgetConfig:
        """Build the settings from the suite config dictionary.

        Args:
            config: Parsed ``config.yaml`` contents (may be empty or None).

        Returns:
            EpochBudgetConfig: Validated settings. Extensions are disabled when
            the section is absent.

        Raises:
            TypeError: If the section is not a mapping or a value is not an integer.
            ValueError: If a value is negative.
        """
        section = (config or {}).get(CONFIG_SECTION) or {}
        if not isinstance(section, Mapping):
            raise TypeError(f"'{CONFIG_SECTION}' must be a mapping, got {type(section).__name__}")
        return cls(
            extension_epochs=_non_negative_int(section, "extension_epochs"),
            max_extension_epochs=_non_negative_int(section, "max_extension_epochs"),
        )


def _non_negative_int(section: Mapping[str, Any], key: str) -> int:
    raw = section.get(key, 0)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"'{CONFIG_SECTION}.{key}' must be an integer, got {raw!r}")
    if raw < 0:
        raise ValueError(f"'{CONFIG_SECTION}.{key}' must be non-negative, got {raw}")
    return raw


@dataclass(frozen=True)
class EpochDecision:
    """Outcome of one end-of-epoch policy evaluation."""

    action: EpochAction
    total_epochs: int
    extension_used: int
    message: str
    stop_reason: StopReason | None = None


class EpochBudgetPolicy:
    """Bounded epoch-loop controller.

    Args:
        total_epochs: Configured epoch budget (inclusive last epoch number).
        config: Budget-extension settings.

    Raises:
        ValueError: If ``total_epochs`` is below 1.
    """

    def __init__(self, total_epochs: int, config: EpochBudgetConfig) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        self.total_epochs = int(total_epochs)
        self.config = config
        self.extension_used = 0

    def decide(
        self,
        epoch: int,
        *,
        converged: bool,
        early_stop: bool,
        governor_reports_improvement: bool,
    ) -> EpochDecision:
        """Decide what the loop does after ``epoch`` has completed.

        Args:
            epoch: The epoch that just finished.
            converged: Whether the 4-pillar completion invariant is met.
            early_stop: Whether the governor signalled early stopping.
            governor_reports_improvement: Whether the governor's audit for this
                epoch ran and judged the model still improving (no early stop).

        Returns:
            EpochDecision: The action, the (possibly extended) budget, and a
            human-readable reason.
        """
        if converged:
            return self._stop(StopReason.CONVERGED, f"all completion criteria met at epoch {epoch}")
        if early_stop:
            return self._stop(
                StopReason.EARLY_STOP, f"governor signalled early stop at epoch {epoch}"
            )
        if epoch < self.total_epochs:
            return EpochDecision(
                action=EpochAction.CONTINUE,
                total_epochs=self.total_epochs,
                extension_used=self.extension_used,
                message=f"epoch {epoch}/{self.total_epochs} complete",
            )

        if not self.config.extensions_enabled:
            return self._stop(
                StopReason.BUDGET_REACHED,
                f"epoch budget of {self.total_epochs} reached (budget extensions disabled)",
            )
        remaining = self.config.max_extension_epochs - self.extension_used
        if remaining <= 0:
            return self._stop(
                StopReason.EXTENSION_LIMIT,
                f"epoch budget of {self.total_epochs} reached and extension ceiling of "
                f"{self.config.max_extension_epochs} epochs exhausted",
            )
        if not governor_reports_improvement:
            return self._stop(
                StopReason.BUDGET_REACHED,
                f"epoch budget of {self.total_epochs} reached and the governor "
                "does not report improvement",
            )

        step = min(self.config.extension_epochs, remaining)
        self.total_epochs += step
        self.extension_used += step
        return EpochDecision(
            action=EpochAction.EXTEND,
            total_epochs=self.total_epochs,
            extension_used=self.extension_used,
            message=(
                f"epoch budget reached at epoch {epoch}; governor reports improvement, "
                f"extending by {step} epochs to {self.total_epochs} ({self.extension_used}/"
                f"{self.config.max_extension_epochs} extension epochs used)"
            ),
        )

    def _stop(self, reason: StopReason, message: str) -> EpochDecision:
        return EpochDecision(
            action=EpochAction.STOP,
            total_epochs=self.total_epochs,
            extension_used=self.extension_used,
            message=message,
            stop_reason=reason,
        )
