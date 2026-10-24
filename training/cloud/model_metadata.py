"""Typed schema for ``model_metadata.yaml`` (single metadata source per model).

The YAML holds everything the Kaggle Models page needs: one ``model`` section for the Kaggle
Model and one ``variations`` entry per Kaggle Model Instance. The two ``kaggle_*_payload``
methods render the exact JSON documents consumed by the Kaggle API
(``model-metadata.json`` / ``model-instance-metadata.json`` semantics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training.cloud.variations import VARIATION_BY_KEY, VARIATIONS

METADATA_FILENAME = "model_metadata.yaml"

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PLACEHOLDER_OWNERS = {"", "INSERT_OWNER_SLUG_HERE"}

_HEADER = (
    "---\n"
    "# LemGendary Kaggle Models metadata (single source for the Kaggle model page).\n"
    "# Generated fields are rebuilt on every publish. To keep a manual edit, list its dotted\n"
    "# path under 'hand_edited' (for example: model.title, variations.pytorch/fp32.usage).\n"
)


class MetadataValidationError(ValueError):
    """Raised when a model_metadata.yaml document is missing or has invalid required fields."""

    def __init__(self, model_key: str, errors: list[str]) -> None:
        self.model_key = model_key
        self.errors = errors
        super().__init__(f"model_metadata for '{model_key}' is invalid: " + "; ".join(errors))


@dataclass
class ModelSection:
    """Kaggle Model level metadata."""

    owner: str
    slug: str
    title: str
    subtitle: str
    description: str
    is_private: bool = False
    license: str = "CC BY-NC 4.0"
    keywords: list[str] = field(default_factory=list)
    provenance_sources: list[str] = field(default_factory=list)
    publish_time: str | None = None


@dataclass
class VariationSection:
    """Kaggle Model Instance (variation) metadata."""

    overview: str
    usage: str
    license: str = "CC BY-NC 4.0"
    fine_tunable: bool = False
    training_data: list[str] = field(default_factory=list)
    model_instance_type: str = "Unspecified"
    base_model_instance: str = ""
    external_base_model_url: str = ""
    version_notes: str = ""


@dataclass
class ModelMetadata:
    """Complete publish metadata for one model."""

    model_key: str
    model: ModelSection
    variations: dict[str, VariationSection]
    hand_edited: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ backward compatibility
    def __getitem__(self, item: str) -> Any:
        return self.kaggle_model_payload()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.kaggle_model_payload().get(item, default)

    def keys(self) -> Any:
        return self.kaggle_model_payload().keys()

    # ------------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Return a list of human readable validation errors (empty when valid)."""
        errors: list[str] = []
        m = self.model
        if m.owner in _PLACEHOLDER_OWNERS or not _OWNER_RE.match(m.owner):
            errors.append("model.owner must be a resolved Kaggle username")
        if not _SLUG_RE.match(m.slug):
            errors.append(f"model.slug '{m.slug}' must be lowercase letters, digits and hyphens")
        if not m.title.strip():
            errors.append("model.title is required")
        if not m.subtitle.strip():
            errors.append("model.subtitle is required")
        if not m.description.strip():
            errors.append("model.description is required")
        if m.is_private:
            errors.append("model.is_private must be false (models are published as public)")
        if not m.license.strip():
            errors.append("model.license is required")
        if m.publish_time is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", m.publish_time):
            errors.append("model.publish_time must be YYYY-MM-DD")

        for variation in VARIATIONS:
            v = self.variations.get(variation.key)
            if v is None:
                errors.append(f"variations.{variation.key} is missing")
                continue
            if not v.overview.strip():
                errors.append(f"variations.{variation.key}.overview is required")
            if not v.usage.strip():
                errors.append(f"variations.{variation.key}.usage is required")
            if not v.license.strip():
                errors.append(f"variations.{variation.key}.license is required")
        for key in self.variations:
            if key not in VARIATION_BY_KEY:
                errors.append(f"variations.{key} is not a supported variation")
        return errors

    def ensure_valid(self) -> None:
        """Raise MetadataValidationError when the document is invalid."""
        errors = self.validate()
        if errors:
            raise MetadataValidationError(self.model_key, errors)

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        """Return the YAML document structure."""
        m = self.model
        doc: dict[str, Any] = {
            "model_key": self.model_key,
            "hand_edited": list(self.hand_edited),
            "model": {
                "owner": m.owner,
                "slug": m.slug,
                "title": m.title,
                "subtitle": m.subtitle,
                "description": m.description,
                "is_private": m.is_private,
                "license": m.license,
                "keywords": list(m.keywords),
                "provenance_sources": list(m.provenance_sources),
                "publish_time": m.publish_time,
            },
            "variations": {},
        }
        for variation in VARIATIONS:
            v = self.variations.get(variation.key)
            if v is None:
                continue
            doc["variations"][variation.key] = {
                "overview": v.overview,
                "usage": v.usage,
                "license": v.license,
                "fine_tunable": v.fine_tunable,
                "training_data": list(v.training_data),
                "model_instance_type": v.model_instance_type,
                "base_model_instance": v.base_model_instance,
                "external_base_model_url": v.external_base_model_url,
                "version_notes": v.version_notes,
            }
        return doc

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> ModelMetadata:
        """Build metadata from a parsed YAML document."""
        model_key = str(doc.get("model_key", ""))
        raw_model = doc.get("model") or {}
        model = ModelSection(
            owner=str(raw_model.get("owner", "")),
            slug=str(raw_model.get("slug", "")),
            title=str(raw_model.get("title", "")),
            subtitle=str(raw_model.get("subtitle", "")),
            description=str(raw_model.get("description", "")),
            is_private=bool(raw_model.get("is_private", False)),
            license=str(raw_model.get("license", "CC BY-NC 4.0")),
            keywords=[str(k) for k in raw_model.get("keywords") or []],
            provenance_sources=[str(s) for s in raw_model.get("provenance_sources") or []],
            publish_time=(str(raw_model["publish_time"]) if raw_model.get("publish_time") else None),
        )
        variations: dict[str, VariationSection] = {}
        for key, raw in (doc.get("variations") or {}).items():
            variations[str(key)] = VariationSection(
                overview=str(raw.get("overview", "")),
                usage=str(raw.get("usage", "")),
                license=str(raw.get("license", "CC BY-NC 4.0")),
                fine_tunable=bool(raw.get("fine_tunable", False)),
                training_data=[str(t) for t in raw.get("training_data") or []],
                model_instance_type=str(raw.get("model_instance_type", "Unspecified")),
                base_model_instance=str(raw.get("base_model_instance", "")),
                external_base_model_url=str(raw.get("external_base_model_url", "")),
                version_notes=str(raw.get("version_notes", "")),
            )
        return cls(
            model_key=model_key,
            model=model,
            variations=variations,
            hand_edited=[str(p) for p in doc.get("hand_edited") or []],
        )

    def to_yaml(self) -> str:
        """Serialize to YAML text with literal block style for multi-line strings."""
        body = yaml.dump(
            self.to_dict(),
            Dumper=_LiteralDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
        return _HEADER + body

    def write(self, path: Path) -> Path:
        """Write the YAML document to path and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ModelMetadata:
        """Load a model_metadata.yaml file."""
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(doc, dict):
            raise MetadataValidationError(path.parent.name, ["top-level YAML document must be a mapping"])
        return cls.from_dict(doc)

    # ------------------------------------------------------------------ Kaggle payloads
    def kaggle_model_payload(self) -> dict[str, Any]:
        """Return the document accepted by the Kaggle model create/update API."""
        m = self.model
        payload: dict[str, Any] = {
            "ownerSlug": m.owner,
            "slug": m.slug,
            "title": m.title,
            "subtitle": m.subtitle,
            "isPrivate": m.is_private,
            "description": m.description,
            "licenseName": m.license,
            "keywords": list(m.keywords),
            "provenanceSources": list(m.provenance_sources),
        }
        if m.publish_time:
            payload["publishTime"] = m.publish_time
        return payload

    def kaggle_instance_payload(self, variation_key: str) -> dict[str, Any]:
        """Return the document accepted by the Kaggle model instance create/update API."""
        variation = VARIATION_BY_KEY[variation_key]
        v = self.variations[variation_key]
        return {
            "ownerSlug": self.model.owner,
            "modelSlug": self.model.slug,
            "instanceSlug": variation.instance_slug,
            "framework": variation.framework,
            "overview": v.overview,
            "usage": v.usage,
            "licenseName": v.license,
            "fineTunable": v.fine_tunable,
            "trainingData": list(v.training_data),
            "modelInstanceType": v.model_instance_type,
            "baseModelInstance": v.base_model_instance,
            "externalBaseModelUrl": v.external_base_model_url,
        }

    # ------------------------------------------------------------------ merge support
    def get_path(self, dotted: str) -> Any:
        """Read a value by dotted path (model.title, variations.onnx/fp16.usage)."""
        node: Any = self.to_dict()
        for part in _split_path(dotted):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(dotted)
            node = node[part]
        return node

    def with_path(self, dotted: str, value: Any) -> ModelMetadata:
        """Return a copy with the value at the dotted path replaced."""
        doc = self.to_dict()
        parts = _split_path(dotted)
        node: Any = doc
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(dotted)
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(dotted)
        node[parts[-1]] = value
        return ModelMetadata.from_dict(doc)


def _split_path(dotted: str) -> list[str]:
    """Split a dotted path; variation keys contain '/' but never '.'."""
    return [p for p in dotted.split(".") if p]


class _LiteralDumper(yaml.SafeDumper):
    """YAML dumper that emits multi-line strings in literal block style."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _represent_str)
