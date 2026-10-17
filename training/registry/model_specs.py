"""Model Registry and Specification Service (SSOT Accessor).

Provides typed, cached, and validated access to unified_models_v2.yaml and
canonical_capabilities.yaml in lemgendary-docs, serving as the Single Source of Truth (SSOT).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import yaml

from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.registry")


@dataclass(frozen=True)
class AcademicReference:
    """Academic citation and publication metadata."""

    title: str = ""
    authors: str = ""
    year: int | str = ""
    venue: str = ""
    link: str = ""
    bibtex: str = ""


@dataclass(frozen=True)
class KaggleModelMetadata:
    """Kaggle Models Hub metadata specification."""

    slug: str
    title: str
    subtitle: str
    is_private: bool = False
    license: str = "CC BY-NC 4.0"
    tags: list[str] = field(default_factory=list)
    provenance_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpecification:
    """Authoritative model specification combining architecture, training, and hub metadata."""

    key: str
    status: str
    name: str
    class_name: str
    category: str
    filename: str
    checkpoint: str
    input_size: Any
    batch_size: Any
    learning_rate: float
    loss_fn: str
    dataset_type: str
    architecture_type: str
    description: str
    datasets: list[str] = field(default_factory=list)
    canonical_format: str = "webdataset"
    kaggle_metadata: KaggleModelMetadata | None = None
    academic_reference: AcademicReference | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_binary_name(self) -> str:
        """Standardized binary stem without extension."""
        return f"LemGendary{self.filename}"


class ModelRegistryService:
    """Singleton service providing SSOT model configuration and academic literature."""

    _instance: ClassVar[ModelRegistryService | None] = None
    _models: ClassVar[dict[str, ModelSpecification]] = {}
    _capabilities: ClassVar[dict[str, Any]] = {}

    def __new__(cls) -> ModelRegistryService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load and cache unified_models_v2.yaml and canonical_capabilities.yaml."""
        project_root = get_project_root()
        manifest_path = project_root / "unified_models_v2.yaml"
        if not manifest_path.exists():
            manifest_path = project_root / "unified_models.yaml"

        caps_path = (project_root.parent / "lemgendary-docs" / "canonical_capabilities.yaml").resolve()

        caps_by_id: dict[str, Any] = {}
        if caps_path.exists():
            try:
                raw_caps = yaml.safe_load(caps_path.read_text(encoding="utf-8")) or {}
                ModelRegistryService._capabilities = raw_caps
                for m in raw_caps.get("models", []):
                    caps_by_id[m.get("id", "")] = m
            except Exception as exc:
                logger.warning("Failed to load canonical_capabilities.yaml: %s", exc)

        models_dict: dict[str, ModelSpecification] = {}
        if manifest_path.exists():
            try:
                raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
                for m_key, raw_val in raw_manifest.items():
                    if m_key == "_registry_metadata" or not isinstance(raw_val, dict):
                        continue

                    # 1. Parse Kaggle metadata
                    k_raw = raw_val.get("kaggle_metadata", {})
                    k_meta = KaggleModelMetadata(
                        slug=k_raw.get("slug", m_key.replace("_", "-")),
                        title=k_raw.get("title", raw_val.get("name", m_key)),
                        subtitle=k_raw.get("subtitle", f"Authoritative checkpoint for {m_key}"),
                        is_private=bool(k_raw.get("is_private", False)),
                        license=k_raw.get("license", "CC BY-NC 4.0"),
                        tags=list(k_raw.get("tags", ["deep-learning", "pytorch", "onnx"])),
                        provenance_sources=list(k_raw.get("provenance_sources", [])),
                    )

                    # 2. Parse academic reference (from canonical_capabilities first, then manifest)
                    cap_info = caps_by_id.get(m_key, {})
                    acad_raw = cap_info.get("academic_reference") or raw_val.get("academic_reference", {})
                    acad_ref = None
                    if acad_raw:
                        acad_ref = AcademicReference(
                            title=acad_raw.get("title", ""),
                            authors=acad_raw.get("authors", ""),
                            year=acad_raw.get("year", ""),
                            venue=acad_raw.get("venue", ""),
                            link=acad_raw.get("link", ""),
                            bibtex=acad_raw.get("bibtex", ""),
                        )

                    spec = ModelSpecification(
                        key=m_key,
                        status=raw_val.get("status", "ACTIVE"),
                        name=raw_val.get("name", m_key),
                        class_name=raw_val.get("class_name", m_key),
                        category=raw_val.get("category", "restoration"),
                        filename=raw_val.get("filename", m_key),
                        checkpoint=raw_val.get("checkpoint", f"{m_key}_best.pth"),
                        input_size=raw_val.get("input_size"),
                        batch_size=raw_val.get("batch_size", 16),
                        learning_rate=float(raw_val.get("learning_rate", 0.0001)),
                        loss_fn=raw_val.get("loss_fn", "l1"),
                        dataset_type=raw_val.get("dataset_type", "paired_image"),
                        architecture_type=raw_val.get("architecture_type", "Neural Network"),
                        description=raw_val.get("description", "").strip(),
                        datasets=list(raw_val.get("datasets", [])),
                        canonical_format=raw_val.get("canonical_format", "webdataset"),
                        kaggle_metadata=k_meta,
                        academic_reference=acad_ref,
                        raw_config=raw_val,
                    )
                    models_dict[m_key] = spec
            except Exception as exc:
                logger.error("Failed to load unified_models_v2.yaml: %s", exc)

        ModelRegistryService._models = models_dict
        logger.info("Loaded %d model specifications into ModelRegistryService", len(models_dict))

    def get_model(self, model_key: str) -> ModelSpecification | None:
        """Retrieve specification for a specific model key."""
        return self._models.get(model_key)

    def get_all_models(self) -> dict[str, ModelSpecification]:
        """Retrieve all registered model specifications."""
        return dict(self._models)

    def list_models(self) -> list[ModelSpecification]:
        """List all registered model specifications as a list."""
        return list(self._models.values())

    def list_keys(self) -> list[str]:
        """List all registered model keys."""
        return list(self._models.keys())


def get_model_registry() -> ModelRegistryService:
    """Convenience accessor for ModelRegistryService singleton."""
    return ModelRegistryService()
