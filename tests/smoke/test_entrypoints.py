"""Smoke test suite for LemGendary Model Training Suite entry points."""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestEntrypointsSmoke(unittest.TestCase):
    """Verifies that primary CLI entry points and packages initialize cleanly."""

    def test_import_training_package(self) -> None:
        """Verify training package is importable."""
        import training
        self.assertIsNotNone(training)

    def test_import_models_package(self) -> None:
        """Verify models package is importable."""
        import models
        self.assertIsNotNone(models)

    def test_import_data_package(self) -> None:
        """Verify data package is importable."""
        import data
        self.assertIsNotNone(data)

    def test_import_parallel_package(self) -> None:
        """Verify parallel strategy package is importable."""
        from training import parallel
        self.assertIsNotNone(parallel)

    def test_parallel_cli_list(self) -> None:
        """Verify parallel diagnostic CLI runs successfully."""
        cmd = [sys.executable, "-m", "training.parallel", "list"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("single", res.stdout.lower())

    def test_train_cli_help(self) -> None:
        """Verify train.py entry point displays help without crashing."""
        cmd = [sys.executable, "training/train.py", "--help"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}")
        self.assertIn("--model", res.stdout)


if __name__ == "__main__":
    unittest.main()
