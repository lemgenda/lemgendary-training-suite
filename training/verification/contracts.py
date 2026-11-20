"""Model Family Verification Contracts and Schemas.

Defines the verification contract per model family:
1. Loadability (weights + config)
2. Forward determinism (fixed seed + input -> identical output)
3. Metric thresholds (verification on held-out set / SOTA targets)
4. Export equivalence (FP32/FP16 ONNX vs. PyTorch)
5. Schema conformance of metrics/history files
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from training.telemetry.schemas import TaskFamily


class VerificationStage(str, Enum):
    """Execution stages of model family verification."""

    LOADABILITY = "loadability"
    DETERMINISM = "determinism"
    METRIC_THRESHOLDS = "metric_thresholds"
    EXPORT_EQUIVALENCE = "export_equivalence"
    SCHEMA_CONFORMANCE = "schema_conformance"


@dataclass(frozen=True)
class CheckpointContract:
    """Specification of checkpoint format, minimum size and mandatory payload keys."""

    min_size_bytes: int = 1024
    required_keys: tuple[str, ...] = ("model_state_dict",)
    strict_weights: bool = False


@dataclass(frozen=True)
class DeterminismContract:
    """Specification for reproducibility and forward determinism under fixed seed."""

    seed: int = 42
    atol: float = 1e-6
    rtol: float = 1e-5
    eval_trials: int = 2


@dataclass(frozen=True)
class MetricThresholdContract:
    """Specification of held-out metric expectations and target directionality."""

    required_metrics: tuple[str, ...]
    min_targets: dict[str, float] = field(default_factory=dict)
    max_targets: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportEquivalenceContract:
    """Specification of tolerance bounds for ONNX FP32 and ONNX FP16 vs PyTorch."""

    fp32_atol: float = 1e-3
    fp32_rtol: float = 1e-2
    fp16_atol: float = 1e-2
    fp16_rtol: float = 1e-2


@dataclass(frozen=True)
class SchemaConformanceContract:
    """Specification of required CSV/JSON telemetry headers and format validation."""

    task_family: TaskFamily
    strict_columns: bool = False
    enforce_history: bool = True
    enforce_metrics: bool = True


@dataclass(frozen=True)
class ModelFamilyContract:
    """Contract bundling all verification pillars for a model family."""

    family: TaskFamily
    checkpoint: CheckpointContract
    determinism: DeterminismContract
    metrics: MetricThresholdContract
    export: ExportEquivalenceContract
    schema: SchemaConformanceContract


@dataclass
class StageVerificationResult:
    """Result of a single verification stage."""

    stage: VerificationStage
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ModelVerificationReport:
    """Comprehensive machine-readable verification report."""

    model_key: str
    family: str
    overall_passed: bool
    checkpoint_path: str | None = None
    stages: dict[str, StageVerificationResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        data = asdict(self)
        # Stage results enum serialization
        data["stages"] = {
            k: {
                "stage": v.stage.value if hasattr(v.stage, "value") else str(v.stage),
                "passed": v.passed,
                "details": v.details,
                "error": v.error,
            }
            for k, v in self.stages.items()
        }
        return data
