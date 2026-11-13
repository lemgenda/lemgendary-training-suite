"""Model hub folders are created only for known models, under the configured hub root."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from training.cloud.kaggle_model_sync import KaggleModelSyncManager
from training.cloud.model_metadata import MetadataValidationError
from training.export.staging import VariationStagingService
from training.utils.paths import WORKSPACE_ROOT_ENV


def _tree(root: Path) -> list[str]:
    """Every path below ``root``, relative and sorted."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


class TestHubDirectories(unittest.TestCase):
    """Sync and staging never leave folders for unknown models or outside their hub."""

    def test_unknown_model_metadata_leaves_no_folder(self) -> None:
        """``ensure_model_metadata`` rejects an unknown key before creating its folder."""
        with tempfile.TemporaryDirectory() as hub:
            mgr = KaggleModelSyncManager(username="owner", hub_root=hub)
            with self.assertRaises(MetadataValidationError):
                mgr.ensure_model_metadata("nosuchmodel")
            self.assertEqual(_tree(Path(hub)), [])

    def test_unknown_model_push_leaves_no_folder(self) -> None:
        """``push_model`` validates the key before staging any variation folder."""
        with tempfile.TemporaryDirectory() as hub:
            mgr = KaggleModelSyncManager(username="owner", hub_root=hub)
            with self.assertRaises(MetadataValidationError):
                mgr.push_model("nosuchmodel", dry_run=True)
            self.assertEqual(_tree(Path(hub)), [])
            workspace = Path(os.environ[WORKSPACE_ROOT_ENV])
            self.assertFalse((workspace / "LemGendaryModels").exists())

    def test_dry_run_push_stages_nothing(self) -> None:
        """A dry run validates what is there; it never moves files into dist/kaggle (K-MDL-8)."""
        with tempfile.TemporaryDirectory() as hub:
            mgr = KaggleModelSyncManager(username="owner", hub_root=hub)
            mgr.push_model("mirnet_exposure", dry_run=True)
            self.assertFalse((Path(hub) / "mirnet_exposure" / "dist").exists())
            workspace = Path(os.environ[WORKSPACE_ROOT_ENV])
            self.assertFalse((workspace / "LemGendaryModels").exists())

    def test_default_staging_hub_follows_workspace_root(self) -> None:
        """Staging without ``base_dir`` uses the workspace root's LemGendaryModels."""
        VariationStagingService().stage_variations("mirnet_exposure", validate=False)
        workspace = Path(os.environ[WORKSPACE_ROOT_ENV])
        hub_dir = workspace / "LemGendaryModels" / "mirnet_exposure"
        self.assertTrue((hub_dir / "dist" / "kaggle").is_dir())


if __name__ == "__main__":
    unittest.main()
