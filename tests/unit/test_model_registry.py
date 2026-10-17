"""Unit tests for the Model Registry and Specification Service (Phase 10 / SSOT Integrity).

Validates typed registry access, SSOT integrity, academic citations, Kaggle specs,
and inclusion of all registered models (including ai_helper_lemgend).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from training.registry.model_specs import (
    AcademicReference,
    KaggleModelMetadata,
    ModelRegistryService,
    ModelSpecification,
    get_model_registry,
)
from training.utils.paths import get_project_root


class TestModelRegistry(unittest.TestCase):
    """Test suite for ModelRegistryService and unified_models_v2.yaml."""

    def setUp(self) -> None:
        self.registry = get_model_registry()

    def test_registry_singleton(self) -> None:
        """Verify that get_model_registry returns a singleton instance."""
        reg1 = get_model_registry()
        reg2 = get_model_registry()
        self.assertIs(reg1, reg2)

    def test_minimum_model_count(self) -> None:
        """Verify that all production models are loaded (at least 23 models)."""
        models = self.registry.get_all_models()
        self.assertGreaterEqual(len(models), 23, f"Expected >= 23 models, found {len(models)}")

    def test_ai_helper_lemgend_registered(self) -> None:
        """Verify that ai_helper_lemgend is registered as a first-class model."""
        model = self.registry.get_model("ai_helper_lemgend")
        self.assertIsNotNone(model, "ai_helper_lemgend should be registered in unified_models_v2.yaml")
        assert model is not None
        self.assertEqual(model.key, "ai_helper_lemgend")
        self.assertEqual(model.category, "llm")
        self.assertIn("aihelper", model.filename.lower())
        self.assertIsNotNone(model.kaggle_metadata)
        assert model.kaggle_metadata is not None
        self.assertEqual(model.kaggle_metadata.license, "CC BY-NC 4.0")

    def test_kaggle_metadata_specifications(self) -> None:
        """Verify that all models have non-empty, compliant KaggleModelMetadata."""
        models = self.registry.get_all_models()
        for key, spec in models.items():
            with self.subTest(model_key=key):
                self.assertIsNotNone(spec.kaggle_metadata, f"Model {key} missing kaggle_metadata")
                meta = spec.kaggle_metadata
                assert meta is not None
                self.assertTrue(bool(meta.slug), f"Model {key} has empty slug")
                self.assertTrue(bool(meta.title), f"Model {key} has empty title")
                self.assertTrue(bool(meta.subtitle), f"Model {key} has empty subtitle")
                self.assertEqual(meta.license, "CC BY-NC 4.0")
                self.assertIsInstance(meta.tags, list)
                self.assertGreater(len(meta.tags), 0, f"Model {key} has no tags")

    def test_canonical_binary_names(self) -> None:
        """Verify canonical binary stem format LemGendary{filename}."""
        models = self.registry.get_all_models()
        for key, spec in models.items():
            with self.subTest(model_key=key):
                stem = spec.canonical_binary_name
                self.assertTrue(stem.startswith("LemGendary"), f"Binary stem {stem} should start with 'LemGendary'")
                self.assertNotIn(" ", stem, f"Binary stem {stem} should not contain spaces")

    def test_academic_citations_integrity(self) -> None:
        """Verify academic references are populated for vision and audio models."""
        models = self.registry.get_all_models()
        cited_count = sum(1 for m in models.values() if m.academic_reference is not None)
        self.assertGreater(cited_count, 10, "Expected at least 10 models with academic citations")

    def test_list_keys_matches_get_all_models(self) -> None:
        """Verify list_keys matches the keys from get_all_models."""
        keys = self.registry.list_keys()
        all_models = self.registry.get_all_models()
        self.assertEqual(set(keys), set(all_models.keys()))


if __name__ == "__main__":
    unittest.main()
