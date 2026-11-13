"""Typed response models for the Desktop GUI aggregation endpoints.

Every model entry of ``GET /api/gui/models/with-stats`` carries
``training_status`` and ``is_fully_trained``; both are required in the
schema, so ``openapi.json`` documents them as always present. They are
computed in one place, :func:`training.server.routes.gui_status.training_verdict`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TrainingStatus = Literal[
    "fully_trained", "partially_trained", "weights_ready", "initializing", "unknown"
]
"""Training state of a model.

``unknown`` is used only when the stats for a model could not be computed
(for example a malformed registry entry). The GUI then shows no verdict
instead of a made-up one.
"""

LadderType = Literal["spatial", "timeframe", "context_tokens"]


class SotaMetricDetail(BaseModel):
    """One SOTA target of a model and the best value reached so far."""

    key: str
    label: str
    target: float | str | None
    achieved: float | None
    lower_is_better: bool
    passed: bool


class BestMetricDetail(BaseModel):
    """Best value of a tracked metric, with its SOTA target when one exists."""

    key: str
    label: str
    value: float | str | None
    target: float | str | None
    lower_is_better: bool
    passed: bool


class ModelWithStats(BaseModel):
    """One model of ``GET /api/gui/models/with-stats``.

    ``training_status`` and ``is_fully_trained`` have no default: every entry
    sets them, also when its stats could not be computed (``stats_error``).
    """

    model_key: str
    training_status: TrainingStatus = Field(
        ..., description="Training state from the 4-pillar verdict"
    )
    is_fully_trained: bool = Field(..., description="True only when all 4 completion pillars hold")
    stats_error: str | None = Field(
        None, description="Why the stats could not be computed; status is then 'unknown'"
    )
    name: str
    key: str
    display_name: str
    category: str = "general"
    class_name: str | None = None
    checkpoints_count: int = 0
    has_best_checkpoint: bool = False
    best_checkpoint_size_mb: float | None = None
    best_checkpoint_epoch: int | None = None
    latest_checkpoint_epoch: int | None = None
    resolution: int | list[int] | None = None
    architecture: str = "PyTorch Model"
    task_type: str = "general"
    canonical_format: str | None = None
    parameters_m: float | None = Field(
        default=None,
        description="Parameter count in millions from the registry; null when unknown",
    )
    spatial_ladder: list[int] = Field(default_factory=list)
    ladder_type: LadderType = "spatial"
    is_forex: bool = False
    is_llm: bool = False
    ladder_passed: bool = False
    active_res: int | None = None
    max_res_completed: int | None = None
    target_res: int | None = None
    active_fold: int | None = None
    total_folds: int | None = None
    fold_years_label: str | None = None
    data_fraction_completed: float = 0.0
    data_fraction_passed: bool = False
    checkpoint_exists: bool = False
    pytorch_ready: bool = Field(
        default=False, description="The PyTorch FP32 binary (<Stem>.pt) exists in the model hub"
    )
    onnx_fp32_ready: bool = Field(
        default=False, description="The ONNX FP32 graph (<Stem>_FP32.onnx) exists in the model hub"
    )
    onnx_fp16_ready: bool = Field(
        default=False, description="The ONNX FP16 graph (<Stem>.onnx) exists in the model hub"
    )
    preferred_parallel: str = "single"
    epochs_completed: int = 0
    best_metric: float | str | None = None
    metric_name: str = "Metric"
    sota_target: float | str | None = None
    sota_targets_total: int = 0
    sota_targets_met: int = 0
    sota_all_met: bool = False
    sota_details: list[SotaMetricDetail] = Field(default_factory=list)
    sota_targets_provisional: bool = Field(
        default=False,
        description=(
            "True when the registry marks the SOTA targets provisional: no cited "
            "benchmark backs them yet"
        ),
    )
    best_metrics_details: list[BestMetricDetail] = Field(default_factory=list)
    sota_reached: bool = False
    status: str = "VALIDATED"
    authoritative_status: str = "VALIDATED"
    learning_rate: float | None = None
    batch_size: int | None = None
    default_epochs: int | None = None
    description: str = ""
