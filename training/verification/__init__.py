"""Model verification package exports (SSOT)."""

from training.verification.contracts import (
    CheckpointContract,
    DeterminismContract,
    ExportEquivalenceContract,
    MetricThresholdContract,
    ModelFamilyContract,
    ModelVerificationReport,
    SchemaConformanceContract,
    StageVerificationResult,
    VerificationStage,
)
from training.verification.engine import ModelVerificationEngine
from training.verification.registry import (
    FAMILY_CONTRACTS,
    get_verification_contract,
    resolve_model_family,
)

__all__ = [
    "FAMILY_CONTRACTS",
    "CheckpointContract",
    "DeterminismContract",
    "ExportEquivalenceContract",
    "MetricThresholdContract",
    "ModelFamilyContract",
    "ModelVerificationEngine",
    "ModelVerificationReport",
    "SchemaConformanceContract",
    "StageVerificationResult",
    "VerificationStage",
    "get_verification_contract",
    "resolve_model_family",
]
