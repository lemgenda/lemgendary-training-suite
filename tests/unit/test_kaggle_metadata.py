"""Unit tests for KaggleMetadataService and model-metadata.json serialization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.cloud.metadata_generator import KaggleMetadataService
from training.registry import get_model_registry


class TestKaggleMetadata(unittest.TestCase):
    """Verifies Kaggle model-metadata.json generation against schema contracts."""

    def setUp(self) -> None:
        self.service = KaggleMetadataService(owner_slug="lemtrada")
        self.registry = get_model_registry()

    def test_generate_metadata_schema_keys(self) -> None:
        """Verify all mandatory Kaggle Models Hub schema keys are present."""
        metadata = self.service.generate_metadata("mirnet_exposure")

        expected_keys = {
            "ownerSlug",
            "slug",
            "title",
            "subtitle",
            "isPrivate",
            "description",
            "licenseName",
            "keywords",
            "provenanceSources",
        }
        self.assertTrue(expected_keys.issubset(metadata.keys()))

        # Check public license and visibility contracts
        self.assertEqual(metadata["ownerSlug"], "lemtrada")
        self.assertFalse(metadata["isPrivate"])
        self.assertEqual(metadata["licenseName"], "CC BY-NC 4.0")
        self.assertIsInstance(metadata["keywords"], list)
        self.assertGreater(len(metadata["keywords"]), 0)

    def test_generate_metadata_ai_helper(self) -> None:
        """Verify metadata generation for ai_helper_lemgend."""
        metadata = self.service.generate_metadata("ai_helper_lemgend")

        self.assertFalse(metadata["isPrivate"])
        self.assertEqual(metadata["licenseName"], "CC BY-NC 4.0")
        self.assertEqual(metadata["slug"], "ai-helper-lemgend")
        self.assertIn("causal-lm", metadata["keywords"])
        self.assertIn("pytorch", metadata["keywords"])

    def test_write_metadata_file_persistence(self) -> None:
        """Verify write_metadata persists clean JSON to target output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = self.service.write_metadata("nafnet_debluring", output_dir=tmp_dir)

            self.assertTrue(out_file.exists())
            self.assertEqual(out_file.name, "model_metadata.yaml")

            # Check that model-metadata.json is also written for Kaggle compatibility
            json_file = Path(tmp_dir) / "model-metadata.json"
            self.assertTrue(json_file.exists())

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["slug"], "nafnet-debluring")
            self.assertFalse(data["isPrivate"])
            self.assertEqual(data["licenseName"], "CC BY-NC 4.0")


if __name__ == "__main__":
    unittest.main()
