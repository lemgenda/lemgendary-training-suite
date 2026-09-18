"""Safe subprocess runner with timeouts and error handling for LemGendary Model Training Suite."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("lemtrain.subprocess")


class SubprocessExecutionError(RuntimeError):
    """Raised when an external subprocess terminates with non-zero exit code or times out."""

    def __init__(self, message: str, exit_code: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


def run_command(
    cmd: list[str],
    cwd: Path | str | None = None,
    timeout: float = 60.0,
    check: bool = False,
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run an external command with timeout and structured diagnostic logging.
    
    Raises:
        SubprocessExecutionError: If command times out or returns non-zero when check=True.
    """
    cwd_path = str(Path(cwd).resolve()) if cwd else os.getcwd()
    cmd_str = " ".join(cmd)

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        res = subprocess.run(
            cmd,
            cwd=cwd_path,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False,
            env=full_env,
        )
    except subprocess.TimeoutExpired as exc:
        err_msg = f"Command timed out after {timeout}s: '{cmd_str}' in '{cwd_path}'"
        logger.error(err_msg)
        raise SubprocessExecutionError(err_msg, exit_code=None, stderr="") from exc
    except OSError as exc:
        err_msg = f"Operating system error executing command '{cmd_str}': {exc}"
        logger.error(err_msg)
        raise SubprocessExecutionError(err_msg, exit_code=None, stderr="") from exc

    if check and res.returncode != 0:
        err_msg = f"Command failed with exit code {res.returncode}: '{cmd_str}'\nStderr: {res.stderr}"
        logger.error(err_msg)
        raise SubprocessExecutionError(err_msg, exit_code=res.returncode, stderr=res.stderr)

    return res
