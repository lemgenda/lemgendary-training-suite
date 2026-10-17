"""Kaggle Models Hub Metadata Service.

Generates authoritative model-metadata.json manifests for Kaggle Models Hub publishing,
enforcing Public visibility (isPrivate=False), CC BY-NC 4.0 licensing, and SSOT alignment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from training.registry import ModelSpecification, get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.cloud.metadata")

DEFAULT_OWNER_SLUG = "lemtrada"
DEFAULT_LICENSE = "CC BY-NC 4.0"


class KaggleMetadataService:
    """Generates and writes Kaggle Models Hub model-metadata.json specifications."""

    def __init__(self, owner_slug: str = DEFAULT_OWNER_SLUG) -> None:
        self.owner_slug = owner_slug
        self.registry = get_model_registry()

    def generate_metadata(
        self,
        model_key: str,
        spec: ModelSpecification | None = None,
        owner_slug: str | None = None,
    ) -> dict[str, Any]:
        """Build dictionary representation of Kaggle model-metadata.json."""
        resolved_spec = spec or self.registry.get_model(model_key)
        owner = owner_slug or self.owner_slug

        if resolved_spec and resolved_spec.kaggle_metadata:
            k_meta = resolved_spec.kaggle_metadata
            slug = k_meta.slug
            title = k_meta.title
            subtitle = k_meta.subtitle
            is_private = k_meta.is_private
            license_name = k_meta.license
            keywords = list(k_meta.tags)
            sources = list(k_meta.provenance_sources)
            desc = resolved_spec.description or subtitle
        else:
            slug = model_key.replace("_", "-").lower()
            title = model_key.replace("_", " ").title()
            subtitle = f"Authoritative production weights for {model_key}"
            is_private = False
            license_name = DEFAULT_LICENSE
            keywords = ["deep-learning", "pytorch", "onnx"]
            sources = []
            desc = subtitle

        metadata_dict: dict[str, Any] = {
            "ownerSlug": owner,
            "slug": slug,
            "title": title,
            "subtitle": subtitle,
            "isPrivate": is_private,
            "description": desc,
            "licenseName": license_name,
            "keywords": keywords,
            "provenanceSources": sources,
        }

        return metadata_dict

    def write_metadata(
        self,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        owner_slug: str | None = None,
    ) -> Path:
        """Write model-metadata.json to target destination directory."""
        if output_dir:
            dest_dir = Path(output_dir).resolve()
        else:
            root = get_project_root()
            dest_dir = (root.parent / "LemGendaryModels" / model_key).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        metadata_content = self.generate_metadata(model_key=model_key, spec=spec, owner_slug=owner_slug)
        out_file = dest_dir / "model-metadata.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=2)

        logger.info("Successfully generated Kaggle metadata: %s", out_file)
        return out_file

    def generate_all(
        self,
        base_dir: Path | str | None = None,
        owner_slug: str | None = None,
    ) -> dict[str, Path]:
        """Generate model-metadata.json for all registered models."""
        results: dict[str, Path] = {}
        all_models = self.registry.get_all_models()

        for key, spec in all_models.items():
            if key.startswith("_"):
                continue
            if base_dir:
                m_dir = Path(base_dir).resolve() / key
            else:
                root = get_project_root()
                m_dir = (root.parent / "LemGendaryModels" / key).resolve()

            out_path = self.write_metadata(
                model_key=key,
                output_dir=m_dir,
                spec=spec,
                owner_slug=owner_slug,
            )
            results[key] = out_path

        logger.info("Completed metadata generation for %d models.", len(results))
        return results


def generate_kaggle_metadata(
    model_key: str,
    output_dir: Path | str | None = None,
    owner_slug: str = DEFAULT_OWNER_SLUG,
) -> Path:
    """Convenience functional interface for generating model metadata."""
    svc = KaggleMetadataService(owner_slug=owner_slug)
    return svc.write_metadata(model_key=model_key, output_dir=output_dir)
