"""Model registry inspection and topology audit endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
import yaml

from training.services.audit_service import AuditService
from training.utils.paths import get_project_root

router = APIRouter(prefix="/models", tags=["models"])


def _load_registry(root: Path | None = None) -> dict[str, Any]:
    project_root = root or get_project_root()
    cfg_path = project_root / "config.yaml"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    reg_name = cfg.get("unified_models", "unified_models_v2.yaml")
    reg_path = project_root / reg_name
    if not reg_path.exists():
        return {}

    with open(reg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models_dict = data.get("models")
    if isinstance(models_dict, dict):
        return models_dict
    return {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}


def _get_root(request: Request) -> Path:
    if hasattr(request.app.state, "server_state"):
        return request.app.state.server_state.project_root
    return get_project_root()


@router.get("")
def list_models(request: Request) -> list[dict[str, Any]]:
    """List all registered model architectures."""
    root = _get_root(request)
    registry = _load_registry(root)
    results: list[dict[str, Any]] = []
    for k, info in registry.items():
        results.append({
            "model_key": k,
            "name": info.get("name", k),
            "class_name": info.get("class_name"),
            "category": info.get("category", "general"),
            "resolution": info.get("resolution"),
            "batch_size": info.get("batch_size"),
            "checkpoint": info.get("checkpoint"),
            "status": info.get("status", "VALIDATED"),
        })
    return sorted(results, key=lambda x: x["model_key"])


@router.get("/{model_key}")
def get_model_info(model_key: str, request: Request) -> dict[str, Any]:
    """Retrieve full configuration and metadata for a model."""
    root = _get_root(request)
    registry = _load_registry(root)
    if model_key not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")
    return registry[model_key]


@router.get("/{model_key}/audit")
def audit_model(model_key: str, request: Request) -> dict[str, Any]:
    """Audit parameter counts, layer breakdown, and memory footprint of model."""
    root = _get_root(request)
    service = AuditService(project_root=root)
    try:
        return service.audit_model(model_key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed auditing model: {e}")


@router.post("/{model_key}/docs/generate")
def generate_model_docs(model_key: str, request: Request) -> dict[str, Any]:
    """Regenerate authoritative README.md and Kaggle model-metadata.json for model."""
    root = _get_root(request)
    registry = _load_registry(root)
    if model_key not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")

    try:
        from training.cloud.metadata_generator import KaggleMetadataService
        from training.doc_generator import ModelDocumentationService

        doc_service = ModelDocumentationService()
        meta_service = KaggleMetadataService()

        model_hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()
        model_hub_dir.mkdir(parents=True, exist_ok=True)

        readme_file = doc_service.write_readme(model_key, output_dir=model_hub_dir)
        metadata_file = meta_service.write_metadata(model_key, output_dir=model_hub_dir)

        return {
            "status": "success",
            "model_key": model_key,
            "readme_path": str(readme_file),
            "metadata_path": str(metadata_file),
            "message": f"Documentation and Kaggle metadata regenerated for {model_key}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed generating docs for '{model_key}': {e}")


@router.post("/{model_key}/export/stage")
def stage_model_variations(model_key: str, request: Request) -> dict[str, Any]:
    """Stage exported model binaries into canonical isolated Kaggle variation subdirectories."""
    root = _get_root(request)
    registry = _load_registry(root)
    if model_key not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")

    try:
        from training.export.staging import VariationStagingService

        staging_service = VariationStagingService()
        result = staging_service.stage_variations(model_key=model_key)

        return {
            "status": "success" if result.success else "warning",
            "model_key": model_key,
            "staged_files": {k: str(v) for k, v in result.staged_files.items()},
            "is_valid": result.report.is_valid if result.report else True,
            "errors": result.errors,
            "message": f"Staged variations for {model_key}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed staging variations for '{model_key}': {e}")


@router.get("/{model_key}/variations/validate")
def validate_model_variations(model_key: str, request: Request) -> dict[str, Any]:
    """Validate strict binary isolation across all Kaggle variation directories."""
    root = _get_root(request)
    registry = _load_registry(root)
    if model_key not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in registry")

    try:
        from training.export.validation import StrictIsolationValidator

        validator = StrictIsolationValidator()
        report = validator.validate_all(model_key=model_key)

        return {
            "model_key": model_key,
            "is_valid": report.is_valid,
            "variations": {
                k: {
                    "variation": v.variation,
                    "is_valid": v.is_valid,
                    "files_found": v.files_found,
                    "errors": v.errors,
                    "warnings": v.warnings,
                }
                for k, v in report.variations.items()
            },
            "all_errors": report.all_errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed validating variations for '{model_key}': {e}")
