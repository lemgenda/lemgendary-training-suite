"""Unit tests for LemGendary Canonical Typer CLI (Gate 11).

Verifies CLI command dispatch, error handling, help documentation, and presets via CliRunner.
"""

from __future__ import annotations

import unittest
from typer.testing import CliRunner

from training.cli.lemtrain import app

runner = CliRunner()


class TestLemTrainCLI(unittest.TestCase):
    """Test suite for Typer CLI interface."""

    def test_cli_help(self) -> None:
        """Verify root help displays all command groups."""
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("train", result.output)
        self.assertIn("eval", result.output)
        self.assertIn("export", result.output)
        self.assertIn("notebooks", result.output)
        self.assertIn("checkpoints", result.output)
        self.assertIn("presets", result.output)
        self.assertIn("audit", result.output)
        self.assertIn("sync", result.output)
        self.assertIn("version", result.output)

    def test_cli_version(self) -> None:
        """Verify version command."""
        result = runner.invoke(app, ["version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("LemGendary Model Training Suite v2026.11.0", result.output)

    def test_cli_presets_list(self) -> None:
        """Verify presets list outputs registered profiles."""
        result = runner.invoke(app, ["presets", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("quick-sota", result.output)
        self.assertIn("debug-tiny", result.output)
        self.assertIn("walk-forward", result.output)

    def test_cli_presets_show(self) -> None:
        """Verify presets show returns valid JSON configuration."""
        result = runner.invoke(app, ["presets", "show", "quick-sota"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"epochs": 50', result.output)

    def test_cli_presets_show_invalid(self) -> None:
        """Verify error on invalid preset name."""
        result = runner.invoke(app, ["presets", "show", "non_existent_preset"])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_audit_system(self) -> None:
        """Verify audit system command."""
        result = runner.invoke(app, ["audit", "system"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("System Resource Telemetry", result.output)
        self.assertIn("Python:", result.output)
        self.assertIn("Disk Free:", result.output)

    def test_cli_audit_model(self) -> None:
        """Verify audit model command."""
        result = runner.invoke(app, ["audit", "model", "forex_predictor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model Topology Audit: forex_predictor", result.output)
        self.assertIn("Total Parameters:", result.output)

    def test_cli_checkpoints_list_empty_or_valid(self) -> None:
        """Verify checkpoints list executes without crash."""
        result = runner.invoke(app, ["checkpoints", "list"])
        self.assertEqual(result.exit_code, 0)

    def test_cli_sync_probe(self) -> None:
        """Verify sync probe command outputs telemetry."""
        result = runner.invoke(app, ["sync", "probe"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cloud Provider Health Telemetry", result.output)
        self.assertIn("[GDRIVE]", result.output)
        self.assertIn("[KAGGLE]", result.output)
        self.assertIn("[GITHUB]", result.output)


if __name__ == "__main__":
    unittest.main()
