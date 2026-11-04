"""Tests for the shared ecosystem loader and the facts derived from it."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import ecosystem
from training.cloud.metadata_generator import resolve_whitepaper_url
from training.ecosystem import (
    DATASETS_SERVICE,
    ECOSYSTEM_FILE_ENV,
    ENV_MANAGER_SERVICE,
    TRAINING_SERVICE,
    EcosystemFileError,
)
from training.registry.kaggle_datasets import (
    model_kaggle_dataset_handles,
    model_kaggle_dataset_urls,
    with_kaggle_dataset_refs,
)
from training.utils.dataset_delegate import DatasetCompilerDelegate
from training.utils.env_delegate import EnvManagerDelegate
from training.utils.paths import WORKSPACE_ROOT_ENV, get_workspace_root

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "ecosystem.json"


class TestEcosystemLoader(unittest.TestCase):
    """The loader reads the file named by LEMGENDARY_ECOSYSTEM_FILE (the test fixture)."""

    def setUp(self) -> None:
        ecosystem.clear_cache()

    def tearDown(self) -> None:
        ecosystem.clear_cache()

    def test_reads_fixture(self) -> None:
        """Ports, owner, docs URL and bind host come from the fixture file."""
        self.assertEqual(ecosystem.ecosystem_file(), FIXTURE)
        self.assertEqual(ecosystem.service_port(ENV_MANAGER_SERVICE), 18000)
        self.assertEqual(ecosystem.service_port(DATASETS_SERVICE), 18100)
        self.assertEqual(ecosystem.service_port("training"), 18200)
        self.assertEqual(ecosystem.service_url(TRAINING_SERVICE), "http://127.0.0.1:18200")
        self.assertEqual(ecosystem.kaggle_owner(), "fixture-owner")
        self.assertEqual(ecosystem.docs_url(), "https://docs.fixture.example/hub")
        self.assertEqual(ecosystem.bind_host(), "127.0.0.1")

    def test_unknown_service(self) -> None:
        """An id that is neither a service id nor an alias raises."""
        with self.assertRaises(EcosystemFileError):
            ecosystem.service_port("no-such-service")

    def test_missing_file_names_expected_path(self) -> None:
        """A missing file raises EcosystemFileError that names the expected path."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.json"
            with (
                mock.patch.dict(os.environ, {ECOSYSTEM_FILE_ENV: str(missing)}),
                self.assertRaises(EcosystemFileError) as ctx,
            ):
                ecosystem.ecosystem()
        self.assertIn(str(missing), str(ctx.exception))

    def test_invalid_file(self) -> None:
        """A file with a wrongly typed field is rejected."""
        bad = json.loads(FIXTURE.read_text(encoding="utf-8"))
        bad["services"][0]["port"] = "18000"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ecosystem.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {ECOSYSTEM_FILE_ENV: str(path)}),
                self.assertRaises(EcosystemFileError),
            ):
                ecosystem.ecosystem()

    def test_workspace_root_env(self) -> None:
        """LEMGENDARY_WORKSPACE_ROOT moves the workspace root and the default file location."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {WORKSPACE_ROOT_ENV: tmp}
            with mock.patch.dict(os.environ, env):
                os.environ.pop(ECOSYSTEM_FILE_ENV, None)
                root = Path(tmp).resolve()
                self.assertEqual(get_workspace_root(), root)
                self.assertEqual(
                    ecosystem.ecosystem_file(),
                    root / "lemgendary-env-manager" / "ecosystem" / "ecosystem.json",
                )

    def test_delegates_use_service_urls(self) -> None:
        """The sibling delegates address the sidecars on the file's ports."""
        self.assertEqual(EnvManagerDelegate().sidecar_url, "http://127.0.0.1:18000")
        self.assertEqual(DatasetCompilerDelegate().sidecar_url, "http://127.0.0.1:18100")

    def test_whitepaper_fallback_uses_docs_url(self) -> None:
        """Without a matching docs manifest entry the docs hub index is returned."""
        url = resolve_whitepaper_url("no_such_model_key")
        self.assertEqual(url, "https://docs.fixture.example/hub/index.html")


class TestKaggleDatasetRefs(unittest.TestCase):
    """Kaggle dataset handles derive from the manifolds and the ecosystem owner."""

    def test_derived_from_manifolds(self) -> None:
        """Unique manifolds map to ``<owner>/<lower-cased manifold>``."""
        info = {"datasets": ["LemGendizedUpnV2", "LemGendizedUpnV2"]}
        self.assertEqual(model_kaggle_dataset_handles(info), ["fixture-owner/lemgendizedupnv2"])
        self.assertEqual(
            model_kaggle_dataset_urls(info),
            ["https://www.kaggle.com/datasets/fixture-owner/lemgendizedupnv2"],
        )
        filled = with_kaggle_dataset_refs(info)
        self.assertEqual(filled["kaggle_ref"], "kaggle://fixture-owner/lemgendizedupnv2")

    def test_explicit_override_wins(self) -> None:
        """A per-model ``kaggle_ref`` override is kept as-is."""
        info = {"datasets": ["LemGendizedUpnV2"], "kaggle_ref": "kaggle://other/custom-slug"}
        self.assertEqual(model_kaggle_dataset_handles(info), ["other/custom-slug"])

    def test_no_manifolds(self) -> None:
        """An entry without manifolds has no handles and is returned unchanged."""
        self.assertEqual(model_kaggle_dataset_handles({}), [])
        self.assertEqual(with_kaggle_dataset_refs({"name": "x"}), {"name": "x"})


if __name__ == "__main__":
    unittest.main()
