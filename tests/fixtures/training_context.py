"""Building blocks for hand-made ``TrainingContext`` objects in engine tests."""

from typing import Any

from training.checkpoint import MetricVault
from training.governance import SmartTrainingGovernor, SotaTracker
from training.training import GovernanceComponents


def fresh_governance(model_info: dict[str, Any]) -> GovernanceComponents:
    """Return a new governor, SOTA tracker and metric vault for ``model_info``."""
    return GovernanceComponents(
        governor=SmartTrainingGovernor(model_info=model_info),
        sota_tracker=SotaTracker(),
        vault=MetricVault(),
    )
