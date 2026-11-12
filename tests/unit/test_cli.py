"""Unit tests for LemGendary Canonical Typer CLI (Gate 11).

Verifies CLI command dispatch, error handling, help documentation, and presets via CliRunner.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from typer.testing import CliRunner

from training.cli.lemtrain import app
from training.model_soup import SoupCheckpointsMissing, SoupResult
from training.version import __version__

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
        self.assertIn(f"LemGendary Model Training Suite v{__version__}", result.output)

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

    def test_cli_soup_calls_shared_service(self) -> None:
        """Verify `soup` delegates to training.model_soup.blend_model_soup with parsed options."""
        result_obj = SoupResult(
            "m",
            [Path("vault_psnr.pth"), Path("vault_lpips.pth")],
            [0.6, 0.4],
            Path("best_soup.pth"),
        )
        with mock.patch("training.model_soup.blend_model_soup", return_value=result_obj) as svc:
            result = runner.invoke(
                app, ["soup", "m", "--donors", "psnr,lpips", "--weights", "0.6,0.4"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(svc.call_args.kwargs["recipe"].donors, ["psnr", "lpips"])
        self.assertEqual(svc.call_args.kwargs["recipe"].weights, [0.6, 0.4])

    def test_cli_soup_missing_checkpoints_exit_1(self) -> None:
        """Verify `soup` exits 1 when the service reports missing checkpoints."""
        err = SoupCheckpointsMissing(
            "Missing vault checkpoints: vault_lpips.pth", ["vault_lpips.pth"]
        )
        with mock.patch("training.model_soup.blend_model_soup", side_effect=err):
            result = runner.invoke(app, ["soup", "m"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("vault_lpips.pth", result.output)

    def test_cli_train_passes_option_group(self) -> None:
        """Verify `train` options reach run_training as the TrainRequest keywords."""
        summary = SimpleNamespace(
            model_name="m", status="done", final_epoch=2, total_time=1.0, best_metrics={}
        )
        with (
            mock.patch("training.cli.lemtrain.announce_sidecar"),
            mock.patch("training.cli.lemtrain.run_training", return_value=summary) as run,
        ):
            result = runner.invoke(
                app, ["train", "m", "-e", "2", "--lr", "0.01", "--clean", "--parallel", "dp"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once_with(
            "m",
            epochs=2,
            batch_size=None,
            lr=0.01,
            preset=None,
            env="local",
            clean=True,
            auto_sync=False,
            parallel="dp",
        )

    def test_cli_llm_train_passes_option_group(self) -> None:
        """Verify `llm train` options reach run_training without a parallel strategy."""
        summary = SimpleNamespace(status="done", final_epoch=1)
        with (
            mock.patch("training.cli.llm_commands.announce_sidecar"),
            mock.patch("training.cli.llm_commands.run_training", return_value=summary) as run,
        ):
            result = runner.invoke(app, ["llm", "train", "--auto-sync", "-b", "4"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.args, ("ai_helper_lemgend",))
        self.assertEqual(run.call_args.kwargs["batch_size"], 4)
        self.assertTrue(run.call_args.kwargs["auto_sync"])
        self.assertNotIn("parallel", run.call_args.kwargs)

    def test_cli_audit_judicial_requires_paths_and_builds_request(self) -> None:
        """Verify `audit judicial` keeps its required options and builds the request."""
        missing = runner.invoke(app, ["audit", "judicial", "-m", "a.pth"])
        self.assertNotEqual(missing.exit_code, 0)

        report = {"samples_evaluated": 3, "backend": "cpu", "metrics": {"PLCC": 1, "SRCC": 1}}
        with mock.patch("training.cli.lemtrain.AuditService") as service:
            service.return_value.audit_judicial.return_value = report
            result = runner.invoke(
                app,
                [
                    "audit",
                    "judicial",
                    "-m",
                    "a.pth",
                    "-d",
                    "imgs",
                    "-l",
                    "l.csv",
                    "--device",
                    "cuda",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        request = service.return_value.audit_judicial.call_args.args[0]
        self.assertEqual(request.model_path, "a.pth")
        self.assertEqual(request.device, "cuda")
        self.assertEqual(request.batch_size, 32)

    def test_cli_sync_push_passes_option_group(self) -> None:
        """Verify `sync push` builds the sync manager and push options from its flags."""
        with (
            mock.patch("training.cli.sync_commands.KaggleModelSyncManager") as mgr,
            mock.patch("training.cli.sync_commands._stage_and_push") as push,
        ):
            push.return_value = SimpleNamespace(success=True, variations={})
            result = runner.invoke(
                app, ["sync", "push", "m", "--dry-run", "-v", "onnx/fp16", "-u", "me"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        mgr.assert_called_once_with(username="me", hub_root=None)
        self.assertEqual(
            push.call_args.args[2],
            {"variation": "onnx/fp16", "version_notes": None, "dry_run": True},
        )

    def test_cli_sync_push_rejects_legacy_variation_names(self) -> None:
        """Verify `sync push -v pytorch` (a pre-v16.9.177 name) is rejected before any push."""
        with mock.patch("training.cli.sync_commands.KaggleModelSyncManager") as mgr:
            result = runner.invoke(app, ["sync", "push", "m", "-v", "pytorch"])
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("pytorch/default", str(result.exception) + result.output)
        mgr.assert_not_called()

    def test_cli_sync_push_without_model_or_all_exits_1(self) -> None:
        """Verify `sync push` with neither a model nor --all exits 1."""
        result = runner.invoke(app, ["sync", "push"])
        self.assertEqual(result.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
