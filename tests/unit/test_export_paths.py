"""Unit tests for canonical export paths and binary stem resolution."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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
            hub_dir = staging_service._resolve_hub_dir("mirnet_exposure", base_dir=base_dir)

            self.assertEqual(hub_dir, base_dir.resolve())
            self.assertTrue(hub_dir.exists())

            # Verify stem resolution from SSOT registry
            stem = staging_service._resolve_stem("mirnet_exposure")
            spec = self.registry.get_model("mirnet_exposure")
            self.assertIsNotNone(spec)
            self.assertEqual(stem, spec.filename)

    def test_ai_helper_export_stem_resolution(self) -> None:
        """Verify ai_helper_lemgend resolves to canonical stem AIHelperLemGend."""
        staging_service = VariationStagingService()
        stem = staging_service._resolve_stem("ai_helper_lemgend")
        self.assertEqual(stem, "AIHelperLemGend")


if __name__ == "__main__":
    unittest.main()
