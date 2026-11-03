"""Kaggle Models Hub metadata generator.

Builds ``model_metadata.yaml`` for every model from the registry (``unified_models_v2.yaml``) and
``canonical_capabilities.yaml``. The Kaggle owner is always resolved from the active credentials,
never hardcoded. Regeneration rebuilds generated fields and preserves any field listed under
``hand_edited`` in the existing file.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import yaml

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.model_metadata import (
    METADATA_FILENAME,
    MetadataValidationError,
    ModelMetadata,
    ModelSection,
    VariationSection,
)
from training.cloud.variations import (
    ONNX_FP16,
    PYTORCH_DEFAULT,
    PYTORCH_FP32,
    VARIATIONS,
    KaggleVariation,
)
from training.ecosystem import docs_url
from training.registry import ModelSpecification, get_model_registry
from training.registry.kaggle_datasets import model_kaggle_dataset_urls
from training.utils.paths import get_workspace_root

logger = logging.getLogger("lemtrain.cloud.metadata")

DEFAULT_LICENSE = "CC BY-NC 4.0"
_DEFAULT_SUBTITLE_PREFIX = "Authoritative checkpoint for"


def resolve_whitepaper_url(model_key: str) -> str:
    """Resolve the published GitHub Pages HTML URL for a given model's technical whitepaper."""
    try:
        docs_dir = (get_workspace_root() / "lemgendary-docs").resolve()
        caps_path = docs_dir / "canonical_capabilities.yaml"
        manifest_path = docs_dir / "manifest" / "documents.json"

        if caps_path.exists() and manifest_path.exists():
            caps = yaml.safe_load(caps_path.read_text(encoding="utf-8")) or {}
            models = caps.get("models", [])
            target_model = next((m for m in models if m.get("id") == model_key), None)

            if target_model:
                whitepaper_ref = target_model.get("evidence", {}).get("whitepaper", "")
                md_filename = Path(whitepaper_ref).name

                docs_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for doc in docs_manifest.get("documents", []):
                    doc_md_filename = Path(doc.get("md_path", "")).name
                    if doc_md_filename == md_filename or (
                        model_key == "ai_helper_lemgend" and doc.get("id") == "foundation-llm"
                    ):
                        return f"{docs_url()}/{doc.get('html_path')}"
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        yaml.YAMLError,
        AttributeError,
        TypeError,
    ) as exc:
        # OSError: unreadable docs files; ValueError: json.JSONDecodeError;
        # Attribute/TypeError: capability/manifest entries of unexpected shape.
        logger.warning("Could not resolve whitepaper URL for %s: %s", model_key, exc)

    return f"{docs_url()}/index.html"


class KaggleMetadataService:
    """Generates, merges and writes ``model_metadata.yaml`` files."""

    def __init__(self, owner_slug: str | None = None) -> None:
        self._owner_slug = owner_slug
        self.registry = get_model_registry()

    @property
    def owner_slug(self) -> str:
        """Kaggle owner slug, resolved from credentials on first use."""
        if not self._owner_slug:
            user, _key = resolve_kaggle_credentials()
            self._owner_slug = user
        return self._owner_slug

    # ------------------------------------------------------------------ generation
    def generate_metadata(
        self,
        model_key: str,
        spec: ModelSpecification | None = None,
        owner_slug: str | None = None,
    ) -> ModelMetadata:
        """Build the metadata document for a model from registry data."""
        resolved = spec or self.registry.get_model(model_key)
        if resolved is None:
            raise MetadataValidationError(
                model_key, [f"model '{model_key}' is not in the registry"]
            )

        owner = owner_slug or self.owner_slug
        k_meta = resolved.kaggle_metadata
        slug = k_meta.slug if k_meta and k_meta.slug else model_key.replace("_", "-").lower()
        title = k_meta.title if k_meta and k_meta.title else resolved.name
        subtitle = k_meta.subtitle if k_meta and k_meta.subtitle else ""
        if not subtitle or subtitle.startswith(_DEFAULT_SUBTITLE_PREFIX):
            subtitle = resolved.architecture_type
        license_name = k_meta.license if k_meta and k_meta.license else DEFAULT_LICENSE
        keywords = list(k_meta.tags) if k_meta else []
        sources = list(k_meta.provenance_sources) if k_meta else []
        ref = resolved.academic_reference
        if ref and ref.link and ref.link not in sources:
            sources.append(ref.link)

        whitepaper_url = resolve_whitepaper_url(model_key)
        if whitepaper_url and whitepaper_url not in sources:
            sources.append(whitepaper_url)

        dataset_urls = self._dataset_urls(resolved)
        stem = resolved.filename or model_key

        model = ModelSection(
            owner=owner,
            slug=slug,
            title=title,
            subtitle=subtitle,
            description=self._build_description(
                resolved, (owner, slug), dataset_urls, whitepaper_url, keywords
            ),
            is_private=False,
            license=license_name,
            keywords=keywords,
            provenance_sources=sources,
            publish_time=None,
        )

        variations: dict[str, VariationSection] = {}
        base_handle = f"{owner}/{slug}/pytorch/default"
        for variation in VARIATIONS:
            is_base = variation is PYTORCH_DEFAULT
            instance_type = "BaseModel" if is_base else "KaggleVariant"
            base_inst = "" if is_base else base_handle

            variations[variation.key] = VariationSection(
                overview=self._variation_overview(resolved, variation),
                usage=self._variation_usage(owner, slug, stem, variation),
                license=license_name,
                fine_tunable=is_base,
                training_data=list(dataset_urls),
                model_instance_type=instance_type,
                base_model_instance=base_inst,
                external_base_model_url="",
                version_notes=f"{resolved.name} - {variation.label} release",
            )

        return ModelMetadata(model_key=model_key, model=model, variations=variations)

    def merge_hand_edited(self, generated: ModelMetadata, existing: ModelMetadata) -> ModelMetadata:
        """Keep manual edits: each dotted path in existing.hand_edited overrides generation."""
        merged = generated
        kept: list[str] = []
        for dotted in existing.hand_edited:
            try:
                value = existing.get_path(dotted)
                merged = merged.with_path(dotted, value)
            except KeyError:
                logger.warning(
                    "hand_edited path '%s' no longer exists for %s; dropped",
                    dotted,
                    generated.model_key,
                )
                continue
            kept.append(dotted)
        merged.hand_edited = kept
        return merged

    # ------------------------------------------------------------------ persistence
    def metadata_path(self, model_key: str, output_dir: Path | str | None = None) -> Path:
        """Return the model_metadata.yaml path for a model."""
        if output_dir:
            base = Path(output_dir).resolve()
        else:
            base = (get_workspace_root() / "LemGendaryModels" / model_key).resolve()
        return base / METADATA_FILENAME

    def build(
        self,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        owner_slug: str | None = None,
    ) -> ModelMetadata:
        """Generate metadata, merge hand edits from an existing file and validate."""
        generated = self.generate_metadata(model_key, spec=spec, owner_slug=owner_slug)
        path = self.metadata_path(model_key, output_dir)
        merged = generated
        if path.exists():
            merged = self.merge_hand_edited(generated, ModelMetadata.load(path))
        merged.ensure_valid()
        return merged

    def write_metadata(
        self,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        owner_slug: str | None = None,
    ) -> Path:
        """Generate, merge, validate and write model_metadata.yaml and model-metadata.json.

        Returns the yaml path.
        """
        metadata = self.build(model_key, output_dir=output_dir, spec=spec, owner_slug=owner_slug)
        yaml_path = self.metadata_path(model_key, output_dir)
        metadata.write(yaml_path)

        # Write canonical model-metadata.json in same folder
        json_path = yaml_path.parent / "model-metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata.kaggle_model_payload(), f, indent=2)

        logger.info("Wrote Kaggle model metadata: %s and %s", yaml_path, json_path)
        return yaml_path

    def generate_all(
        self,
        base_dir: Path | str | None = None,
        owner_slug: str | None = None,
    ) -> tuple[dict[str, Path], dict[str, list[str]]]:
        """Write metadata for all registered models; return (written paths, errors per model)."""
        written: dict[str, Path] = {}
        failures: dict[str, list[str]] = {}
        for key, spec in self.registry.get_all_models().items():
            if key.startswith("_"):
                continue
            out_dir = (Path(base_dir).resolve() / key) if base_dir else None
            try:
                written[key] = self.write_metadata(
                    key, output_dir=out_dir, spec=spec, owner_slug=owner_slug
                )
            except MetadataValidationError as exc:
                failures[key] = exc.errors
                logger.error("Metadata generation failed for %s: %s", key, exc)
        logger.info(
            "Metadata generation complete: %d written, %d failed", len(written), len(failures)
        )
        return written, failures

    # ------------------------------------------------------------------ content builders
    @staticmethod
    def _dataset_urls(spec: ModelSpecification) -> list[str]:
        return model_kaggle_dataset_urls(spec.raw_config)

    @staticmethod
    def _build_description(
        spec: ModelSpecification,
        owner_slug: tuple[str, str],
        dataset_urls: list[str],
        whitepaper_url: str | None = None,
        keywords: list[str] | None = None,
    ) -> str:
        """Return the Kaggle model description; ``owner_slug`` is ``(owner, slug)``."""
        owner, slug = owner_slug
        raw_desc = spec.description or spec.name
        wrapped_desc = textwrap.fill(raw_desc, width=120) if len(raw_desc) > 120 else raw_desc
        lines: list[str] = [wrapped_desc, ""]
        lines.append(f"**Architecture:** {spec.architecture_type}")
        lines.append(f"**Category:** {spec.category}")
        active_keywords: list[str] = []
        if keywords:
            active_keywords = list(keywords)
        elif spec.kaggle_metadata and spec.kaggle_metadata.tags:
            active_keywords = list(spec.kaggle_metadata.tags)
        if active_keywords:
            lines.append(f"**Keywords / Tags:** {', '.join(active_keywords)}")
        if dataset_urls:
            lines.append("")
            lines.append("**Training data:**")
            for url in dataset_urls:
                lines.append(f"- {url}")
        lines.append("")
        lines.append("**Variations:**")
        for variation in VARIATIONS:
            lines.append(f"- `{variation.handle(owner, slug)}` - {variation.label}")

        # Citations & Academic / Technical Whitepaper References
        resolved_whitepaper = whitepaper_url or resolve_whitepaper_url(spec.key)
        lines.append("")
        lines.append("### Citations & Technical Documentation")
        if resolved_whitepaper:
            lines.append(
                f"- **Architecture Whitepaper:** [Whitepaper Reference]({resolved_whitepaper})"
            )

        ref = spec.academic_reference
        if ref and ref.title:
            citation = f"{ref.authors} ({ref.year}). {ref.title}. {ref.venue}".strip()
            full_ref = f"- **Academic Reference:** {citation} {ref.link}".strip()
            if len(full_ref) > 120:
                wrapped_ref = textwrap.fill(
                    full_ref, width=120, break_long_words=False, break_on_hyphens=False
                )
                lines.append(wrapped_ref)
            else:
                lines.append(full_ref)
        elif not resolved_whitepaper:
            hub = docs_url()
            lines.append(f"- **Documentation Hub:** [{hub}]({hub})")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _variation_overview(spec: ModelSpecification, variation: KaggleVariation) -> str:
        stem = spec.filename or spec.key
        if variation is PYTORCH_DEFAULT:
            return (
                f"Training checkpoints for {spec.name} (best.pth, latest.pth) with the model "
                "README and "
                "training metrics.csv. Includes complete fine-tuning state and architecture "
                "telemetry for evaluation, "
                "resuming training, or transfer learning."
            )
        if variation is PYTORCH_FP32:
            return (
                f"Production PyTorch FP32 standalone TorchScript weight binary ({stem}.pt) for "
                f"{spec.name}. "
                "Optimized for direct CPU and CUDA server inference without external dependencies."
            )
        if variation is ONNX_FP16:
            return (
                f"ONNX half-precision (FP16) computational graph ({stem}.onnx) for {spec.name}. "
                "Engineered for high-throughput, low-latency, and low-VRAM edge inference via ONNX "
                "Runtime with TensorRT or CUDA Execution Providers."
            )
        return (
            f"ONNX full-precision (FP32) computational graph ({stem}_FP32.onnx) for {spec.name}. "
            "Engineered for high-precision validation and broad hardware compatibility. "
            "Note: The .onnx graph and its external weights tensor file (.onnx.data) must reside "
            "in the same runtime directory."
        )

    @staticmethod
    def _variation_usage(owner: str, slug: str, stem: str, variation: KaggleVariation) -> str:
        handle = variation.handle(owner, slug)
        if variation is PYTORCH_DEFAULT:
            return (
                "```python\n"
                "import kagglehub\n"
                "import torch\n\n"
                f'path = kagglehub.model_download("{handle}")\n'
                'checkpoint = torch.load(f"{path}/checkpoints/best.pth", map_location="cpu")\n'
                "```\n"
            )
        if variation is PYTORCH_FP32:
            return (
                "```python\n"
                "import kagglehub\n"
                "import torch\n\n"
                f'path = kagglehub.model_download("{handle}")\n'
                f'weights = torch.load(f"{{path}}/{stem}.pt", map_location="cpu")\n'
                "```\n"
            )
        file_name = f"{stem}.onnx" if variation is ONNX_FP16 else f"{stem}_FP32.onnx"
        return (
            "```python\n"
            "import kagglehub\n"
            "import onnxruntime as ort\n\n"
            f'path = kagglehub.model_download("{handle}")\n'
            "session = ort.InferenceSession(\n"
            f'    f"{{path}}/{file_name}",\n'
            '    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],\n'
            ")\n"
            "```\n"
        )


def load_model_metadata(model_key: str, base_dir: Path | str | None = None) -> ModelMetadata:
    """Load and validate the existing model_metadata.yaml for a model."""
    svc = KaggleMetadataService()
    path = svc.metadata_path(model_key, Path(base_dir) / model_key if base_dir else None)
    metadata = ModelMetadata.load(path)
    metadata.ensure_valid()
    return metadata


def generate_kaggle_metadata(
    model_key: str,
    output_dir: Path | str | None = None,
    owner_slug: str | None = None,
) -> Path:
    """Convenience functional interface for generating model_metadata.yaml."""
    svc = KaggleMetadataService(owner_slug=owner_slug)
    return svc.write_metadata(model_key=model_key, output_dir=output_dir)


__all__ = [
    "KaggleMetadataService",
    "generate_kaggle_metadata",
    "load_model_metadata",
]
