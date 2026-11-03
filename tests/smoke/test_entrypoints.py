"""Smoke test suite for LemGendary Model Training Suite entry points."""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestEntrypointsSmoke(unittest.TestCase):
    """Verifies that primary CLI entry points and packages initialize cleanly."""

    def _assert_importable(self, module: str) -> None:
        """Assert ``module`` imports cleanly in a fresh interpreter (no sys.modules cache)."""
        res = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"import {module} failed: {res.stderr}")

    def test_import_training_package(self) -> None:
        """Verify training package is importable."""
        self._assert_importable("training")

    def test_import_models_package(self) -> None:
        """Verify models package is importable."""
        self._assert_importable("models")

    def test_import_data_package(self) -> None:
        """Verify data package is importable."""
        self._assert_importable("data")

    def test_import_parallel_package(self) -> None:
        """Verify parallel strategy package is importable."""
        self._assert_importable("training.parallel")

    def test_parallel_cli_list(self) -> None:
        """Verify parallel diagnostic CLI runs successfully."""
        cmd = [sys.executable, "-m", "training.parallel", "list"]
        res = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("single", res.stdout.lower())

    def test_doc_generator_cli_help(self) -> None:
        """Verify doc_generator entry point displays help without crashing."""
        cmd = [sys.executable, "-m", "training.doc_generator", "--help"]
        res = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("--model", res.stdout)

    def test_kaggle_cloud_manager_cli_help(self) -> None:
        """Verify kaggle_cloud_manager entry point displays help without crashing."""
        cmd = [sys.executable, "-m", "training.kaggle_cloud_manager", "--help"]
        res = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("--action", res.stdout)

    def test_notebook_generator_cli_help(self) -> None:
        """Verify notebook_generator entry point displays help without crashing."""
        cmd = [sys.executable, "-m", "training.notebook_generator", "--help"]
        res = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("--model", res.stdout)


if __name__ == "__main__":
    unittest.main()
