"""Pre-configured Model Family Verification Contracts (SSOT)."""

from __future__ import annotations

from training.registry import ModelSpecification, get_model_registry
from training.telemetry.schemas import TaskFamily
from training.verification.contracts import (
    CheckpointContract,
    DeterminismContract,
    ExportEquivalenceContract,
    MetricThresholdContract,
    ModelFamilyContract,
    SchemaConformanceContract,
)

FAMILY_CONTRACTS: dict[TaskFamily, ModelFamilyContract] = {
    TaskFamily.VISION: ModelFamilyContract(
        family=TaskFamily.VISION,
        checkpoint=CheckpointContract(
            min_size_bytes=1024,
            required_keys=("model_state_dict",),
        ),
        determinism=DeterminismContract(seed=42, atol=1e-6, rtol=1e-5),
        metrics=MetricThresholdContract(
            required_metrics=("psnr", "ssim"),
            min_targets={"psnr": 20.0, "ssim": 0.70},
            max_targets={"lpips": 0.40, "fid": 50.0},
        ),
        export=ExportEquivalenceContract(
            fp32_atol=1e-3, fp32_rtol=1e-2, fp16_atol=1e-2, fp16_rtol=1e-2
        ),
        schema=SchemaConformanceContract(
            task_family=TaskFamily.VISION,
            strict_columns=False,
            enforce_history=True,
            enforce_metrics=True,
        ),
    ),
    TaskFamily.FOREX: ModelFamilyContract(
        family=TaskFamily.FOREX,
        checkpoint=CheckpointContract(
            min_size_bytes=1024,
            required_keys=("model_state_dict",),
        ),
        determinism=DeterminismContract(seed=42, atol=1e-6, rtol=1e-5),
        metrics=MetricThresholdContract(
            required_metrics=("dir_acc", "win_rate"),
            min_targets={"dir_acc": 50.0, "win_rate": 50.0, "profit_factor": 1.0},
            max_targets={"max_drawdown": 30.0},
        ),
        export=ExportEquivalenceContract(
            fp32_atol=1e-4, fp32_rtol=1e-3, fp16_atol=1e-3, fp16_rtol=1e-2
        ),
        schema=SchemaConformanceContract(
            task_family=TaskFamily.FOREX,
            strict_columns=False,
            enforce_history=True,
            enforce_metrics=True,
        ),
    ),
    TaskFamily.YOLO: ModelFamilyContract(
        family=TaskFamily.YOLO,
        checkpoint=CheckpointContract(
            min_size_bytes=1024,
            required_keys=("model_state_dict",),
        ),
        determinism=DeterminismContract(seed=42, atol=1e-5, rtol=1e-4),
        metrics=MetricThresholdContract(
            required_metrics=("map50",),
            min_targets={"map50": 0.50},
        ),
        export=ExportEquivalenceContract(
            fp32_atol=1e-3, fp32_rtol=1e-2, fp16_atol=1e-2, fp16_rtol=1e-2
        ),
        schema=SchemaConformanceContract(
            task_family=TaskFamily.YOLO,
            strict_columns=False,
            enforce_history=True,
            enforce_metrics=True,
        ),
    ),
    TaskFamily.FACE_DETECTION: ModelFamilyContract(
        family=TaskFamily.FACE_DETECTION,
        checkpoint=CheckpointContract(
            min_size_bytes=1024,
            required_keys=("model_state_dict",),
        ),
        determinism=DeterminismContract(seed=42, atol=1e-5, rtol=1e-4),
        metrics=MetricThresholdContract(
            required_metrics=("ap50",),
            min_targets={"ap50": 0.50},
        ),
        export=ExportEquivalenceContract(
            fp32_atol=1e-3, fp32_rtol=1e-2, fp16_atol=1e-2, fp16_rtol=1e-2
        ),
        schema=SchemaConformanceContract(
            task_family=TaskFamily.FACE_DETECTION,
            strict_columns=False,
            enforce_history=True,
            enforce_metrics=True,
        ),
    ),
    TaskFamily.LLM: ModelFamilyContract(
        family=TaskFamily.LLM,
        checkpoint=CheckpointContract(
            min_size_bytes=1024,
            required_keys=("model_state_dict",),
        ),
        determinism=DeterminismContract(seed=42, atol=1e-5, rtol=1e-4),
        metrics=MetricThresholdContract(
            required_metrics=("eval_loss",),
            max_targets={"eval_loss": 5.0, "perplexity": 100.0},
        ),
        export=ExportEquivalenceContract(
            fp32_atol=1e-3, fp32_rtol=1e-2, fp16_atol=1e-2, fp16_rtol=1e-2
        ),
        schema=SchemaConformanceContract(
            task_family=TaskFamily.LLM,
            strict_columns=False,
            enforce_history=True,
            enforce_metrics=True,
        ),
    ),
}


def resolve_model_family(model_key: str, spec: ModelSpecification | None = None) -> TaskFamily:
    """Resolve the canonical TaskFamily for a given model key and optional spec."""
    key_lower = model_key.lower()
    if "forex" in key_lower:
        return TaskFamily.FOREX
    if "yolo" in key_lower:
        return TaskFamily.YOLO
    if "retinaface" in key_lower or "face" in key_lower:
        return TaskFamily.FACE_DETECTION
    if "llm" in key_lower or "ai_helper" in key_lower:
        return TaskFamily.LLM

    if spec:
        cat = (spec.category or "").lower()
        if "forex" in cat:
            return TaskFamily.FOREX
        if "yolo" in cat or "detection" in cat:
            return TaskFamily.YOLO
        if "face" in cat:
            return TaskFamily.FACE_DETECTION
        if "llm" in cat or "nlp" in cat or "causal" in cat:
            return TaskFamily.LLM

    return TaskFamily.VISION


def get_verification_contract(
    model_key: str, spec: ModelSpecification | None = None
) -> ModelFamilyContract:
    """Get the verification contract for a model based on its family and registry targets."""
    if spec is None:
        try:
            reg = get_model_registry()
            spec = reg.get_model(model_key)
        except Exception:
            spec = None

    family = resolve_model_family(model_key, spec)
    base_contract = FAMILY_CONTRACTS[family]

    # If model specification provides explicit sota_targets, overlay them
    if spec and hasattr(spec, "kwargs"):
        pass

    return base_contract
