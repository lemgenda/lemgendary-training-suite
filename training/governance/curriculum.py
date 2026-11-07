"""Curriculum state: resolution ladders, data-fraction expansion, and effective batch scaling."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lemtrain.governance.curriculum")

DEFAULT_SPATIAL_LADDER: tuple[int, ...] = (256, 384, 512)
"""Canonical spatial resolution ladder (px) for vision models."""

DEFAULT_FOREX_LADDER: tuple[int, ...] = (1, 5, 15, 60, 240, 1440)
"""Canonical multi-timeframe curriculum ladder (minutes) for ForexPredictor."""


def is_forex_model(
    model_identifier: str | dict[str, Any] | None = None,
    model_info_or_key: str | dict[str, Any] | None = None,
    *,
    model_info: dict[str, Any] | None = None,
    model_key: str = "",
) -> bool:
    """Return True when the model configuration or key identifies the Forex predictor family.

    Supports positional calls in either order (model_key, model_info) or
    (model_info, model_key), as well as explicit keyword arguments.
    """
    key = model_key
    info = model_info or {}

    for arg in (model_identifier, model_info_or_key):
        if isinstance(arg, str) and not key:
            key = arg
        elif isinstance(arg, dict) and not info:
            info = arg

    return (
        key == "forex_predictor"
        or info.get("category") == "forex"
        or info.get("task_type") == "forex"
        or info.get("dataset_type") == "forex"
        or info.get("filename") == "ForexPredictor"
    )


@dataclass
class CurriculumState:
    """Manage progressive resolution, sample-fraction expansion, and gradient accumulation."""

    res_ladder: list[int] = field(default_factory=lambda: [256])
    current_res: int = 256
    current_fraction: float = 0.15
    fraction_increment: float = 0.15
    current_batch: int = 16
    current_acc: int = 1
    target_effective_batch: int = 24

    def promote_resolution(self) -> bool:
        """Advance resolution to the next available ladder rung. Returns True if promoted."""
        if not self.res_ladder:
            return False

        try:
            curr_idx = self.res_ladder.index(self.current_res)
        except ValueError:
            # If current_res is not exactly on ladder, find next larger rung
            candidates = [r for r in self.res_ladder if r > self.current_res]
            if candidates:
                self.current_res = candidates[0]
                logger.info("Promoted curriculum resolution to %d px", self.current_res)
                return True
            return False

        if curr_idx < len(self.res_ladder) - 1:
            self.current_res = self.res_ladder[curr_idx + 1]
            logger.info(
                "Promoted curriculum resolution: %d -> %d px",
                self.res_ladder[curr_idx],
                self.current_res,
            )
            return True

        return False

    def expand_fraction(self) -> bool:
        """Expand training dataset sample fraction by fraction_increment up to 1.0."""
        if self.current_fraction >= 1.0:
            return False

        new_frac = min(1.0, round(self.current_fraction + self.fraction_increment, 4))
        if new_frac > self.current_fraction:
            old_frac = self.current_fraction
            self.current_fraction = max(self.current_fraction, new_frac)
            logger.info(
                "Expanded curriculum sample fraction: %.1f%% -> %.1f%%",
                old_frac * 100.0,
                new_frac * 100.0,
            )
            return True

        return False

    def throttle_on_oom(self, min_batch: int = 1) -> tuple[int, int]:
        """Halve physical batch size and double gradient accumulation on OOM.

        Preserves target effective batch while drastically dropping peak activation memory.
        Returns (new_batch, new_acc).
        """
        old_batch = self.current_batch
        new_batch = max(min_batch, self.current_batch // 2)
        self.update_batch_and_accumulation(new_batch)
        logger.warning(
            "Curriculum OOM Throttling: batch size reduced %d -> %d, accumulation adjusted to %d "
            "(effective batch=%d)",
            old_batch,
            self.current_batch,
            self.current_acc,
            self.current_batch * self.current_acc,
        )
        return self.current_batch, self.current_acc

    def update_batch_and_accumulation(self, new_batch: int) -> None:
        """Adjust batch size and gradient accumulation to preserve target effective batch."""
        self.current_batch = max(1, new_batch)
        self.current_acc = max(
            1, round(float(self.target_effective_batch) / float(self.current_batch))
        )
        logger.debug(
            "Curriculum batch updated: batch=%d, accum=%d (effective=%d)",
            self.current_batch,
            self.current_acc,
            self.current_batch * self.current_acc,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "res_ladder": list(self.res_ladder),
            "current_res": self.current_res,
            "current_fraction": self.current_fraction,
            "fraction_increment": self.fraction_increment,
            "current_batch": self.current_batch,
            "current_acc": self.current_acc,
            "target_effective_batch": self.target_effective_batch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumState":
        """Deserialize state from dictionary."""
        return cls(
            res_ladder=data.get("res_ladder", [256]),
            current_res=int(data.get("current_res", 256)),
            current_fraction=float(data.get("current_fraction", 0.15)),
            fraction_increment=float(data.get("fraction_increment", 0.15)),
            current_batch=int(data.get("current_batch", 16)),
            current_acc=int(data.get("current_acc", 1)),
            target_effective_batch=int(data.get("target_effective_batch", 24)),
        )
