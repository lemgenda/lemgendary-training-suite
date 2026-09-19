"""Unit and integration tests for LemGendary FastAPI Sidecar Service (Gate 12).

Verifies ServerState, SQLite WAL persistence, JobManager, REST endpoints, and token authentication.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import unittest
from fastapi.testclient import TestClient
import yaml

from training.server.app import create_app
from training.server.jobs import JobManager
from training.server.state import ServerState


class TestServerSubsystem(unittest.TestCase):
    """Test suite for server state, SQLite persistence, and REST endpoints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create dummy presets.yaml
        presets_data = {
            "presets": {
                "test-sota": {
                    "description": "Test SOTA Preset",
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                },
                "quick-sota": {
                    "description": "Quick SOTA profile",
                    "epochs": 10,
                    "batch_size": 8,
                    "learning_rate": 0.0005,
                },
            }
        }
        with open(self.root / "presets.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(presets_data, f)

        # Create dummy config.yaml
        config_data = {
            "unified_models": "unified_models.yaml",
            "training": {"default_epochs": 10},
        }
        with open(self.root / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        # Create dummy unified_models.yaml
        models_data = {
            "test_model": {
                "name": "Test Vision Model",
                "class_name": "TestModelClass",
                "category": "vision",
                "resolution": 256,
                "batch_size": 16,
                "checkpoint": "test.pth",
            }
        }
        with open(self.root / "unified_models.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(models_data, f)

        self.server_state = ServerState(project_root=self.root)

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except OSError:
            pass

    def test_state_db_and_tokens(self) -> None:
        """Verify SQLite DB initialization, WAL mode, token generation, and PID tracking."""
        token = self.server_state.get_or_create_token()
        self.assertTrue(len(token) >= 32)
        self.assertTrue(self.server_state.verify_token(token))
        self.assertFalse(self.server_state.verify_token("invalid_token"))

        self.server_state.write_pid(12345)
        self.assertEqual(self.server_state.read_pid(), 12345)
        self.server_state.clear_pid()
        self.assertIsNone(self.server_state.read_pid())

    def test_job_persistence_lifecycle(self) -> None:
        """Verify job insertion, status transition, metrics update, and recovery."""
        job = self.server_state.insert_job(
            job_id="job_test_001",
            job_type="train",
            model_key="test_model",
            params={"epochs": 5},
        )
        self.assertEqual(job["id"], "job_test_001")
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["params"]["epochs"], 5)

        # Update to running
        self.server_state.update_job_status("job_test_001", status="running")
        running_job = self.server_state.get_job("job_test_001")
        self.assertIsNotNone(running_job)
        if running_job:
            self.assertEqual(running_job["status"], "running")
            self.assertIsNotNone(running_job["started_at"])

        # Update to completed
        metrics = {"val_loss": 0.042, "epochs": 5}
        self.server_state.update_job_status("job_test_001", status="completed", metrics=metrics)
        completed_job = self.server_state.get_job("job_test_001")
        self.assertIsNotNone(completed_job)
        if completed_job:
            self.assertEqual(completed_job["status"], "completed")
            self.assertEqual(completed_job["metrics"]["val_loss"], 0.042)
            self.assertIsNotNone(completed_job["completed_at"])

        # Test orphan recovery
        self.server_state.insert_job("job_orphaned", "train", "test_model")
        self.server_state.update_job_status("job_orphaned", status="running")
        recovered_count = self.server_state.recover_orphaned_jobs()
        self.assertEqual(recovered_count, 1)

        orphaned_job = self.server_state.get_job("job_orphaned")
        self.assertIsNotNone(orphaned_job)
        if orphaned_job:
            self.assertEqual(orphaned_job["status"], "failed")
            self.assertIn("Interrupted", orphaned_job["error_message"])

    def test_rest_api_endpoints(self) -> None:
        """Verify FastAPI REST endpoints for health, presets, models, jobs, and GUI state."""
        app = create_app(project_root=self.root, enforce_auth=False)
        with TestClient(app) as client:
            # 1. Health
            resp = client.get("/api/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "lemgendary-training-suite")
            self.assertEqual(data["port"], 8200)

            # 2. Presets
            resp = client.get("/api/presets")
            self.assertEqual(resp.status_code, 200)
            presets = resp.json()
            self.assertIn("test-sota", presets)

            resp = client.get("/api/presets/test-sota")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["epochs"], 5)

            # 3. Models
            resp = client.get("/api/models")
            self.assertEqual(resp.status_code, 200)
            models = resp.json()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["model_key"], "test_model")

            resp = client.get("/api/models/test_model")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["category"], "vision")

            # 4. Env telemetry
            resp = client.get("/api/env/telemetry")
            self.assertEqual(resp.status_code, 200)
            env_data = resp.json()
            self.assertIn("platform", env_data)
            self.assertIn("python_version", env_data)

            # 5. GUI state
            resp = client.get("/api/gui/state")
            self.assertEqual(resp.status_code, 200)
            gui_state = resp.json()
            self.assertEqual(gui_state["status"], "online")
            self.assertEqual(gui_state["models_count"], 1)

            # 6. Job submission & cancellation
            app.state.job_manager._run_job_worker = lambda *args: None
            job_payload = {"model": "test_model", "epochs": 2, "batch_size": 4}
            resp = client.post("/api/training/train", json=job_payload)
            self.assertEqual(resp.status_code, 200)
            job_id = resp.json()["job_id"]

            resp = client.get(f"/api/jobs/{job_id}")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["id"], job_id)

            resp = client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(resp.status_code, 200)

            resp = client.get(f"/api/jobs/{job_id}/logs")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("lines", resp.json())

    def test_token_authentication(self) -> None:
        """Verify token enforcement when enforce_auth is enabled."""
        app = create_app(project_root=self.root, enforce_auth=True)
        valid_token = self.server_state.get_or_create_token()

        with TestClient(app) as client:
            # Health endpoint is public
            resp = client.get("/api/health")
            self.assertEqual(resp.status_code, 200)

            # Protected endpoint without token returns 401
            resp = client.get("/api/presets")
            self.assertEqual(resp.status_code, 401)

            # Protected endpoint with invalid token returns 401
            resp = client.get("/api/presets", headers={"Authorization": "Bearer invalid"})
            self.assertEqual(resp.status_code, 401)

            # Protected endpoint with valid bearer token returns 200
            resp = client.get("/api/presets", headers={"Authorization": f"Bearer {valid_token}"})
            self.assertEqual(resp.status_code, 200)

            # Protected endpoint with valid custom header returns 200
            resp = client.get("/api/presets", headers={"X-LemTrain-Token": valid_token})
            self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
