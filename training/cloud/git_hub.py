"""Git-LFS model hub synchronizer with atomic rebase loop and stealth masking."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

from training.cloud.credentials import mask_secret, resolve_github_credentials
from training.cloud.manager import CloudManager, CloudSyncError

logger = logging.getLogger("lemtrain.cloud.git_hub")


class GitHubSyncManager:
    """Synchronizes model artifacts, weights, and metrics to GitHub model hub."""

    def __init__(
        self,
        hub_root: Path | None = None,
        hub_user: str = "lemgenda",
        hub_repo: str = "lemgendary-pretrained-models",
        pat: str | None = None,
    ) -> None:
        self.hub_user = hub_user
        self.hub_repo = hub_repo
        self.pat = resolve_github_credentials(pat)

        if hub_root is not None:
            self.hub_root = hub_root.resolve()
        else:
            is_cloud = os.environ.get("KAGGLE_WORKING_DIR") or "/kaggle/working" in str(Path.cwd())
            if is_cloud:
                self.hub_root = Path("/kaggle/working/LemGendaryModels").resolve()
            else:
                self.hub_root = (Path.cwd().parent / "LemGendaryModels").resolve()

    def run_git(self, args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        """Run git command safely with masked secret logging.

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        run_cwd = cwd or self.hub_root
        cmd = ["git"] + args
        try:
            res = subprocess.run(
                cmd,
                cwd=str(run_cwd),
                capture_output=True,
                text=True,
                timeout=120,
            )
            stdout_masked = mask_secret(res.stdout, self.pat)
            stderr_masked = mask_secret(res.stderr, self.pat)
            if res.returncode != 0:
                logger.debug("Git command %s returned %d: %s", args, res.returncode, stderr_masked)
            return res.returncode, stdout_masked, stderr_masked
        except subprocess.TimeoutExpired as exc:
            raise CloudSyncError(
                code="GIT_TIMEOUT",
                message=f"Git command '{' '.join(args)}' timed out after 120s",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise CloudSyncError(
                code="GIT_EXEC_ERROR",
                message=f"Failed to execute git command: {exc}",
                retryable=False,
            ) from exc

    def sync(
        self,
        model_name: str,
        epoch: int,
        src_dir: Path,
        branch: str = "main",
        max_retries: int = 3,
        **kwargs: Any,
    ) -> bool:
        """Stage, commit, and push model artifacts through an atomic rebase loop."""
        if not (self.hub_root / ".git").exists():
            logger.info("Hub repository not initialized at %s. Skipping Git sync.", self.hub_root)
            return False

        is_cloud = bool(os.environ.get("KAGGLE_WORKING_DIR") or "/kaggle/working" in str(self.hub_root))
        if not self.pat and is_cloud:
            logger.warning("GITHUB_PAT missing in cloud environment. Skipping push.")
            return False

        # Metric merge-persistence: fetch remote updates
        metrics_file = self.hub_root / model_name / "metrics.csv"
        if metrics_file.exists():
            self.run_git(["fetch", "origin"])
            self.run_git(["rebase", f"origin/{branch}"])

        # Configure user identities
        self.run_git(["config", "user.email", "lem.treursic@gmail.com"])
        self.run_git(["config", "user.name", self.hub_user])
        self.run_git(["lfs", "install"])
        self.run_git(["add", "."])

        # Check if changes exist
        code, _, _ = self.run_git(["diff-index", "--quiet", "HEAD", "--"])
        if code == 0:
            logger.info("Everything up-to-date in hub. No drift detected.")
            return True

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"SOTA Update: {model_name} | Epoch {epoch} | {timestamp}"
        self.run_git(["commit", "-m", commit_msg])

        for attempt in range(1, max_retries + 1):
            logger.info("Attempting atomic Git push (Try %d/%d)...", attempt, max_retries)
            push_code, _, push_err = self.run_git(["push", "origin", branch])
            if push_code == 0:
                logger.info("Successfully pushed manifold artifacts to GitHub (%s/%s).", self.hub_user, self.hub_repo)
                return True

            logger.warning("Push collision detected: %s. Rebase-syncing...", push_err.strip())
            self.run_git(["pull", "--rebase", "-X", "theirs", "origin", branch])

        raise CloudSyncError(
            code="GIT_PUSH_EXHAUSTED",
            message=f"Exhausted {max_retries} Git push attempts for {model_name}",
            retryable=True,
        )

    def pull(self, remote_identifier: str, dest_dir: Path, **kwargs: Any) -> bool:
        """Pull latest changes from remote Git repository."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        if (dest_dir / ".git").exists():
            code, _, err = self.run_git(["pull", "--ff-only"], cwd=dest_dir)
            if code != 0:
                raise CloudSyncError(
                    code="GIT_PULL_FAILED",
                    message=f"Git pull failed: {err}",
                    retryable=True,
                )
            return True

        clone_url = f"https://github.com/{self.hub_user}/{remote_identifier}.git"
        if self.pat:
            clone_url = f"https://{self.pat}@github.com/{self.hub_user}/{remote_identifier}.git"

        try:
            res = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(dest_dir)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if res.returncode != 0:
                masked_err = mask_secret(res.stderr, self.pat)
                raise CloudSyncError(
                    code="GIT_CLONE_FAILED",
                    message=f"Git clone failed: {masked_err}",
                    retryable=True,
                )
            return True
        except subprocess.TimeoutExpired as exc:
            raise CloudSyncError(
                code="GIT_CLONE_TIMEOUT",
                message="Git clone operation timed out after 180s",
                retryable=True,
            ) from exc

    def probe_health(self) -> dict[str, Any]:
        """Verify Git installation, hub directory existence, and credentials."""
        git_installed = False
        try:
            res = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
            git_installed = (res.returncode == 0)
        except OSError:
            pass

        return {
            "provider": "github",
            "git_installed": git_installed,
            "hub_root_exists": self.hub_root.exists(),
            "has_pat": bool(self.pat),
            "hub_repo": f"{self.hub_user}/{self.hub_repo}",
        }
