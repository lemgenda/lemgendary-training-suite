"""Schema validation engine for LemGendary model configurations.

Validates unified_models_v2.yaml and runtime configuration mappings against
the canonical model schema definition using JSONSchema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from training.utils.paths import get_project_root


class ModelConfigValidationError(ValueError):
    """Raised when a model configuration fails schema validation."""


def load_model_schema() -> dict[str, Any]:
    """Load the canonical model schema JSON definition."""
    schema_path = Path(__file__).resolve().parent / "model_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Model schema not found at {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_model_config(
    config: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> None:
    """Validate a parsed model registry dictionary against the canonical schema.

    Raises:
        ModelConfigValidationError: If any validation checks fail.
    """
    if schema is None:
        schema = load_model_schema()

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: e.path)

    if errors:
        error_msgs = []
        for err in errors:
            path_str = ".".join(str(p) for p in err.path) if err.path else "root"
            error_msgs.append(f"[{path_str}] {err.message}")
        full_report = "\n  - ".join(error_msgs)
        raise ModelConfigValidationError(
            f"Model configuration schema validation failed with {len(errors)} error(s):\n"
            f"  - {full_report}"
        )


def validate_registry_file(file_path: Path | str | None = None) -> None:
    """Validate the canonical unified_models_v2.yaml file directly.

    Args:
        file_path: Optional explicit path; defaults to project root unified_models_v2.yaml.
    """
    if file_path is None:
        file_path = get_project_root() / "unified_models_v2.yaml"
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found at {path}")

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ModelConfigValidationError(
            f"Expected mapping in {path}, got {type(raw_data).__name__}"
        )
    validate_model_config(raw_data)
