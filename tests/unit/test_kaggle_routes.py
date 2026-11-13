"""Kaggle sidecar routes reject unknown model keys with 404 before doing any work."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from training.server.app import create_app
from training.utils.paths import WORKSPACE_ROOT_ENV

_UNKNOWN = "nosuchmodel"
_DETAIL = f"Model '{_UNKNOWN}' not found in registry"


def _tree(root: Path) -> list[str]:
    """Relative paths of everything below ``root``."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


class TestKaggleRoutesUnknownModel(unittest.TestCase):
    """Every Kaggle route taking a model key answers 404 for an unknown key."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.workspace = Path(os.environ[WORKSPACE_ROOT_ENV])

    def _post_unknown(self, path: str, payload: dict) -> None:
        """POST ``payload`` and check the 404, the detail, no job and no files."""
        app = create_app(project_root=self.root, enforce_auth=False)
        with TestClient(app) as client:
            before = _tree(self.root)
            resp = client.post(f"/api/training/kaggle/{path}", json=payload)
            self.assertEqual(resp.status_code, 404, resp.text)
            self.assertEqual(resp.json()["detail"], _DETAIL)
            self.assertEqual(client.get("/api/jobs").json(), [])
            self.assertEqual(_tree(self.root), before)
        self.assertEqual(_tree(self.workspace), [])

    def test_push_model_unknown_is_404(self) -> None:
        """``/kaggle/push-model`` validates the key instead of failing with 500."""
        self._post_unknown("push-model", {"model": _UNKNOWN, "dry_run": True})

    def test_push_unknown_is_404(self) -> None:
        """``/kaggle/push`` enqueues no job for an unknown key."""
        self._post_unknown("push", {"model": _UNKNOWN})

    def test_pull_unknown_is_404(self) -> None:
        """``/kaggle/pull`` enqueues no job for an unknown key."""
        self._post_unknown("pull", {"model": _UNKNOWN})

    def test_train_unknown_is_404(self) -> None:
        """``/kaggle/train`` enqueues no job for an unknown key."""
        self._post_unknown("train", {"model": _UNKNOWN})

    def test_monitor_unknown_model_is_404(self) -> None:
        """``/kaggle/monitor`` rejects an explicit unknown model key."""
        self._post_unknown("monitor", {"kernel_slug": "owner/kernel", "model": _UNKNOWN})


if __name__ == "__main__":
    unittest.main()
