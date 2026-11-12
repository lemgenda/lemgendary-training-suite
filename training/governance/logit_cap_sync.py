"""Apply the governor's logit clamp to models with a configurable logit soft cap (R2-T6).

The governor starts its clamp at the registry ``stabilizers.logit_clamp`` (the value the
model factory gives the model) and adjusts it during training (thermal resets, turbulence
breakouts). Models with ``set_logit_cap`` (NIMA, the EfficientNet classifiers) follow it,
so the ``Clamp`` column of ``metrics.csv`` is the bound the model actually used.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger("lemtrain.governance.logit_cap")


def sync_logit_cap(model: Any, governor: Any) -> bool:
    """Set ``model``'s soft cap to the governor's clamp; return whether it changed."""
    setter = getattr(model, "set_logit_cap", None)
    clamp = getattr(governor, "current_clamp", None)
    if not callable(setter) or not isinstance(clamp, (int, float)) or not math.isfinite(clamp):
        return False
    current = float(getattr(model, "logit_cap", float("nan")))
    if current == float(clamp):
        return False
    setter(float(clamp))
    logger.info("[CLAMP] Logit soft cap %.2f -> %.2f (governor).", current, float(clamp))
    return True
