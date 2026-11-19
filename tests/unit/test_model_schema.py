"""Unit tests for model schema validation (Prompt 20).

Validates:
  1. Valid parses of unified_models_v2.yaml pass completely.
  2. Missing required fields (optimizer, lr_schedule, checkpoint_policy, etc.) fail clearly.
  3. Invalid field types/enums (e.g. invalid category, negative batch size) fail clearly.
  4. Integration with ModelRegistryService preflight validation.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from training.config.schema_validator import (
    ModelConfigValidationError,
    load_model_schema,
    validate_model_config,
    validate_registry_file,
)
from training.registry.model_specs import ModelRegistryService
from training.utils.paths import get_project_root


def _load_canonical_raw_config() -> dict[str, Any]:
    """Helper to load the canonical unified_models_v2.yaml."""
    path = get_project_root() / "unified_models_v2.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_canonical_unified_models_v2_passes_validation() -> None:
    """Verify the canonical unified_models_v2.yaml passes schema validation."""
    validate_registry_file()


def test_schema_loads_correctly() -> None:
    """Verify load_model_schema loads valid json schema."""
    schema = load_model_schema()
    assert isinstance(schema, dict)
    assert schema.get("title") == "LemGendaryModelRegistrySchema"
    assert "_registry_metadata" in schema.get("required", [])


def test_validation_fails_on_missing_registry_metadata() -> None:
    """Verify validation fails if _registry_metadata is missing."""
    cfg = copy.deepcopy(_load_canonical_raw_config())
    del cfg["_registry_metadata"]

    with pytest.raises(ModelConfigValidationError) as exc_info:
        validate_model_config(cfg)
    assert "_registry_metadata" in str(exc_info.value)


def test_validation_fails_on_missing_required_model_field() -> None:
    """Verify validation fails if a mandatory field like optimizer or lr_schedule is deleted."""
    cfg = copy.deepcopy(_load_canonical_raw_config())
    model_entry = cfg["forex_predictor"]
    del model_entry["optimizer"]

    with pytest.raises(ModelConfigValidationError) as exc_info:
        validate_model_config(cfg)
    assert "optimizer" in str(exc_info.value)


def test_validation_fails_on_invalid_category() -> None:
    """Verify validation fails if an invalid category enum is supplied."""
    cfg = copy.deepcopy(_load_canonical_raw_config())
    cfg["forex_predictor"]["category"] = "unsupported_cat"

    with pytest.raises(ModelConfigValidationError) as exc_info:
        validate_model_config(cfg)
    assert "category" in str(exc_info.value)


def test_validation_fails_on_invalid_batch_size() -> None:
    """Verify validation fails if batch_size is negative or an unsupported string."""
    cfg = copy.deepcopy(_load_canonical_raw_config())
    cfg["forex_predictor"]["batch_size"] = -5

    with pytest.raises(ModelConfigValidationError) as exc_info:
        validate_model_config(cfg)
    assert "batch_size" in str(exc_info.value)


def test_model_registry_service_loads_with_schema_validation() -> None:
    """Verify ModelRegistryService loads all 23 models cleanly under schema validation."""
    registry = ModelRegistryService()
    assert len(registry.list_keys()) >= 23
    assert "forex_predictor" in registry.list_keys()
    assert "yolov8n" in registry.list_keys()
