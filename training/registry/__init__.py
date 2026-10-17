"""Model registry package exposing typed model specification services."""

from training.registry.model_specs import (
    AcademicReference,
    KaggleModelMetadata,
    ModelRegistryService,
    ModelSpecification,
    get_model_registry,
)

__all__ = [
    "AcademicReference",
    "KaggleModelMetadata",
    "ModelRegistryService",
    "ModelSpecification",
    "get_model_registry",
]
