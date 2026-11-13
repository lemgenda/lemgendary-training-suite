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


BUDGET_MODEL_INFO: dict[str, Any] = {
    "name": "budget_model",
    "dataset_type": "quality",
    "input_size": 256,
    "batch_size": 12,
    "accumulation_steps": 2,
    "optimization": {"res_ladder": [256, 384], "initial_fraction": 0.2},
}
"""A quality model at micro-batch 12 x accumulation 2 on a 256/384 ladder (batch refits)."""
