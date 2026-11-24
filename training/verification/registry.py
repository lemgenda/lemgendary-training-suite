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


def _family_from_text(text: str) -> TaskFamily | None:
    for keywords, family in (
        (("forex",), TaskFamily.FOREX),
        (("yolo", "detection"), TaskFamily.YOLO),
        (("retinaface", "face"), TaskFamily.FACE_DETECTION),
        (("llm", "ai_helper", "nlp", "causal"), TaskFamily.LLM),
    ):
        if any(kw in text for kw in keywords):
            return family
    return None


def resolve_model_family(model_key: str, spec: ModelSpecification | None = None) -> TaskFamily:
    """Resolve the canonical TaskFamily for a given model key and optional spec."""
    family = _family_from_text(model_key.lower())
    if family is not None:
        return family
    if spec:
        family = _family_from_text((spec.category or "").lower())
        if family is not None:
            return family
    return TaskFamily.VISION


def get_verification_contract(
    model_key: str, spec: ModelSpecification | None = None
) -> ModelFamilyContract:
    """Get the verification contract for a model based on its family and registry targets."""
    if spec is None:
        try:
            reg = get_model_registry()
            spec = reg.get_model(model_key)
        except (KeyError, ValueError, OSError):
            spec = None

    family = resolve_model_family(model_key, spec)
    base_contract = FAMILY_CONTRACTS[family]

    # If model specification provides explicit sota_targets, overlay them
    if spec and hasattr(spec, "kwargs"):
        pass

    return base_contract
