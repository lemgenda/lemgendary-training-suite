"""Delegation adapter for LemGendary Environment Manager."""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from training import ecosystem
from training.utils.paths import get_project_root, get_workspace_root
from training.utils.sidecar import sidecar_online
from training.utils.subprocess import SubprocessExecutionError, run_command

logger = logging.getLogger("lemtrain.env_delegate")


class EnvManagerDelegate:
    """Typed access to the LemGendary Environment Manager CLI and sidecar API (ecosystem port)."""

    def __init__(self, project_root: Path | None = None, sidecar_url: str | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.workspace_root = get_workspace_root()
        self.sidecar_url = sidecar_url or ecosystem.service_url(ecosystem.ENV_MANAGER_SERVICE)

    def is_sidecar_online(self) -> bool:
        """Probe environment manager sidecar API health."""
        return sidecar_online(self.sidecar_url)

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
        except (OSError, ValueError) as exc:
            # OSError covers urllib.error.URLError/HTTPError and socket timeouts;
            # ValueError covers json.JSONDecodeError and UnicodeDecodeError.
            logger.warning("Failed to fetch hardware profile from sidecar: %s", exc)
        return {}

    def validate_codebase(self) -> bool:
        """Run full multi-gate validation on lemgendary-training-suite via lem-env."""
        cmd = [
            "python",
            "-m",
            "env_manager.cli",
            "validate",
            "-p",
            "lemgendary-training-suite",
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
