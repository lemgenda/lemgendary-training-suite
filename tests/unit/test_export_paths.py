"""Unit tests for canonical export paths and binary stem resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.export.common import resolve_export_paths
from training.export.staging import VariationStagingService
from training.registry import get_model_registry


class TestExportPaths(unittest.TestCase):
    """Verifies that model export paths resolve strictly to canonical structures."""

    def setUp(self) -> None:
        self.registry = get_model_registry()

    def test_resolve_export_paths_canonical_stem(self) -> None:
        """Verify resolve_export_paths produces LemGendary{filename} binary stems."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_info = {"filename": "NafNetDeblur"}
            base_name, prod_dir = resolve_export_paths(
                model_key="nafnet_debluring",
                model_info=model_info,
                config={"export_dir": "LemGendaryModels"},
                project_root=root,
            )

            self.assertEqual(base_name, "LemGendaryNafNetDeblur")
            self.assertEqual(prod_dir, (root / "LemGendaryModels" / "nafnet_debluring").resolve())
            self.assertTrue(prod_dir.exists())

    def test_staging_service_stem_and_hub_dir_resolution(self) -> None:
        """Verify VariationStagingService resolves destination variation folders correctly."""
        staging_service = VariationStagingService()
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "LemGendaryModels" / "mirnet_exposure"
            weights = Path(tmp_dir) / "weights.pt"
            weights.write_bytes(b"pt")
            result = staging_service.stage_variations(
                "mirnet_exposure", source_files={"pt": weights}, base_dir=base_dir, validate=False
            )

            hub_dir = base_dir.resolve()
            self.assertTrue(hub_dir.exists())

            # The staged .pt lands in the hub dir under the SSOT registry stem.
            spec = self.registry.get_model("mirnet_exposure")
            self.assertIsNotNone(spec)
            expected = hub_dir / "dist" / "kaggle" / "pytorch" / f"{spec.filename}.pt"
            self.assertEqual(result.staged_files["pytorch"], expected)
            self.assertTrue(expected.exists())

    def test_ai_helper_export_stem_resolution(self) -> None:
        """Verify ai_helper_lemgend resolves to canonical stem AIHelperLemGend."""
        staging_service = VariationStagingService()
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "LemGendaryModels" / "ai_helper_lemgend"
            weights = Path(tmp_dir) / "weights.pt"
            weights.write_bytes(b"pt")
            result = staging_service.stage_variations(
                "ai_helper_lemgend", source_files={"pt": weights}, base_dir=base_dir, validate=False
            )
            self.assertEqual(result.staged_files["pytorch"].name, "AIHelperLemGend.pt")


if __name__ == "__main__":
    unittest.main()
