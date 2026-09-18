"""Delegation adapter for LemGendary Environment Manager."""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from training.utils.paths import get_project_root, get_workspace_root
from training.utils.subprocess import run_command, SubprocessExecutionError

logger = logging.getLogger("lemtrain.env_delegate")


class EnvManagerDelegate:
    """Provides typed access to LemGendary Environment Manager CLI and sidecar API (port 8000)."""

    def __init__(self, project_root: Path | None = None, sidecar_port: int = 8000) -> None:
        self.project_root = project_root or get_project_root()
        self.workspace_root = get_workspace_root()
        self.sidecar_url = f"http://127.0.0.1:{sidecar_port}"

    def is_sidecar_online(self) -> bool:
        """Probe environment manager sidecar API health."""
        try:
            req = urllib.request.Request(f"{self.sidecar_url}/api/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def get_hardware_profile(self) -> dict[str, Any]:
        """Fetch deep hardware profile from sidecar API if online, else empty dict."""
        if not self.is_sidecar_online():
            return {}
        try:
            req = urllib.request.Request(f"{self.sidecar_url}/api/hardware", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, dict):
                        return data
        except Exception as exc:
            logger.warning("Failed to fetch hardware profile from sidecar: %s", exc)
        return {}

    def validate_codebase(self) -> bool:
        """Run full multi-gate validation on lemgendary-training-suite via lem-env."""
        cmd = [
            "python", "-m", "env_manager.cli", "validate",
            "-p", "lemgendary-training-suite",
        ]
        try:
            res = run_command(cmd, cwd=self.workspace_root, timeout=120.0, check=False)
            if res.returncode == 0:
                logger.info("Environment Manager validation succeeded.")
                return True
            logger.error("Environment Manager validation failed:\n%s", res.stdout or res.stderr)
            return False
        except SubprocessExecutionError as exc:
            logger.error("Failed to execute env-manager validation: %s", exc)
            return False
