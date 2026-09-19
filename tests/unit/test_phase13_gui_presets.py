"""Unit tests for Phase 13: Canonical Presets & Desktop GUI Integration (Gate 13).

Verifies:
- Frozen OpenAPI 3.1.0 schema specification contract.
- Desktop GUI state and model stats endpoints.
- Dispatch and parameter persistence of all 5 canonical presets via API.
- Parameter overrides in QuickTrainRequest.
- CLI server openapi export command.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import unittest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from training.cli.lemtrain import app as cli_app
from training.server.app import create_app
from training.server.state import ServerState
from training.utils.paths import get_project_root


class TestPhase13OpenAPIAndContract(unittest.TestCase):
    """Verify OpenAPI 3.1 contract and schema freezing."""

    def setUp(self) -> None:
        self.project_root = get_project_root()
        self.openapi_file = self.project_root / "openapi.json"

    def test_frozen_openapi_contract_exists_and_valid(self) -> None:
        """Verify frozen openapi.json exists, is valid JSON, and declares OpenAPI 3.1."""
        self.assertTrue(self.openapi_file.exists(), "openapi.json must exist at project root")
        content = self.openapi_file.read_text(encoding="utf-8")
        data = json.loads(content)

        self.assertEqual(data.get("openapi"), "3.1.0")
        info = data.get("info", {})
        self.assertEqual(info.get("title"), "LemGendary Model Training Suite Sidecar API")
        self.assertEqual(info.get("version"), "2026.11.0")

        paths = data.get("paths", {})
        required_paths = [
            "/api/health",
            "/api/config",
            "/api/presets",
            "/api/jobs",
            "/api/jobs/{job_id}",
            "/api/jobs/{job_id}/cancel",
            "/api/jobs/{job_id}/logs",
            "/api/models",
            "/api/models/{model_key}",
            "/api/models/{model_key}/audit",
            "/api/training/train",
            "/api/training/evaluate",
            "/api/training/export",
            "/api/datasets",
            "/api/env/telemetry",
            "/api/gui/state",
            "/api/gui/models/with-stats",
            "/api/gui/quick-train",
        ]
        for path in required_paths:
            self.assertIn(path, paths, f"Path '{path}' missing from OpenAPI contract")

    def test_live_app_openapi_matches(self) -> None:
        """Verify live FastAPI application generates 200 OK for /openapi.json."""
        app = create_app(project_root=self.project_root, enforce_auth=False)
        with TestClient(app) as client:
            resp = client.get("/openapi.json")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data.get("openapi"), "3.1.0")
            self.assertIn("/api/gui/quick-train", data.get("paths", {}))


class TestPhase13DesktopGUIAndPresets(unittest.TestCase):
    """Verify Desktop GUI routes and canonical presets execution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        real_root = get_project_root()

        # Copy canonical presets.yaml and unified_models_v2.yaml
        shutil.copy2(real_root / "presets.yaml", self.root / "presets.yaml")
        shutil.copy2(real_root / "config.yaml", self.root / "config.yaml")
        shutil.copy2(real_root / "unified_models_v2.yaml", self.root / "unified_models_v2.yaml")

        self.state = ServerState(project_root=self.root)
        self.app = create_app(project_root=self.root, enforce_auth=False)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        if hasattr(self.app.state, "job_manager"):
            self.app.state.job_manager.shutdown()
        self.client.close()
        try:
            self.temp_dir.cleanup()
        except OSError:
            pass

    def test_gui_state_snapshot(self) -> None:
        """Verify GET /api/gui/state returns unified dashboard snapshot."""
        resp = self.client.get("/api/gui/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["status"], "online")
        self.assertIn("system", data)
        self.assertIn("presets", data)
        self.assertIn("quick-sota", data["presets"])
        self.assertIn("debug-tiny", data["presets"])
        self.assertIn("walk-forward", data["presets"])
        self.assertIn("restoration-ultra", data["presets"])
        self.assertIn("detection-yolo", data["presets"])
        self.assertGreaterEqual(data["models_count"], 0)
        self.assertIn("active_jobs_count", data)
        self.assertIn("pending_jobs_count", data)
        self.assertIn("recent_jobs", data)

    def test_gui_models_with_stats(self) -> None:
        """Verify GET /api/gui/models/with-stats returns models with checkpoint metadata."""
        resp = self.client.get("/api/gui/models/with-stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

        if data:
            first = data[0]
            self.assertIn("model_key", first)
            self.assertIn("name", first)
            self.assertIn("category", first)
            self.assertIn("checkpoints_count", first)
            self.assertIn("has_best_checkpoint", first)

    def test_all_five_canonical_presets_dispatch(self) -> None:
        """Verify all 5 canonical presets dispatch cleanly via POST /api/gui/quick-train."""
        canonical_presets = [
            ("quick-sota", 50, 64, 0.001),
            ("debug-tiny", 2, 4, 0.0001),
            ("walk-forward", 100, 512, 0.0005),
            ("restoration-ultra", 80, 16, 0.0002),
            ("detection-yolo", 100, 32, 0.001),
        ]

        for preset_name, exp_epochs, exp_batch, exp_lr in canonical_presets:
            payload = {
                "model_key": "mirnet_exposure",
                "preset": preset_name,
                "clean": False,
            }
            resp = self.client.post("/api/gui/quick-train", json=payload)
            self.assertEqual(resp.status_code, 200, f"Failed for preset {preset_name}: {resp.text}")
            result = resp.json()

            self.assertIn("job_id", result)
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["preset"], preset_name)

            # Inspect job persisted in SQLite
            job = self.state.get_job(result["job_id"])
            self.assertIsNotNone(job)
            self.assertIn(job["status"], ["pending", "running", "completed", "failed"])
            self.assertEqual(job["model_key"], "mirnet_exposure")
            self.assertEqual(job["job_type"], "train")

            params: dict[str, Any] = job["params"]
            self.assertEqual(params.get("preset"), preset_name)
            self.assertEqual(params.get("epochs"), exp_epochs)
            self.assertEqual(params.get("batch_size"), exp_batch)
            self.assertAlmostEqual(params.get("learning_rate"), exp_lr)

    def test_quick_train_parameter_overrides(self) -> None:
        """Verify QuickTrainRequest supports custom parameter overrides."""
        payload = {
            "model_key": "mirnet_exposure",
            "preset": "quick-sota",
            "epochs": 15,
            "batch_size": 8,
            "learning_rate": 0.0003,
            "env": "kaggle",
            "clean": True,
        }
        resp = self.client.post("/api/gui/quick-train", json=payload)
        self.assertEqual(resp.status_code, 200)
        result = resp.json()

        job = self.state.get_job(result["job_id"])
        self.assertIsNotNone(job)
        params: dict[str, Any] = job["params"]
        self.assertEqual(params.get("epochs"), 15)
        self.assertEqual(params.get("batch_size"), 8)
        self.assertEqual(params.get("learning_rate"), 0.0003)
        self.assertEqual(params.get("env"), "kaggle")
        self.assertTrue(params.get("clean"))

    def test_quick_train_invalid_preset_rejected(self) -> None:
        """Verify invalid preset returns HTTP 400."""
        payload = {
            "model_key": "mirnet_exposure",
            "preset": "nonexistent-preset-xyz",
        }
        resp = self.client.post("/api/gui/quick-train", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid preset", resp.json()["detail"])


class TestPhase13CLIExport(unittest.TestCase):
    """Verify CLI export functionality for OpenAPI schema."""

    def test_cli_server_openapi_export(self) -> None:
        """Verify lemtrain server openapi --output writes valid JSON."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "test_openapi.json"
            result = runner.invoke(cli_app, ["server", "openapi", "--output", str(out_file)])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(out_file.exists())

            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(data.get("openapi"), "3.1.0")
            self.assertIn("/api/health", data.get("paths", {}))


if __name__ == "__main__":
    unittest.main()
