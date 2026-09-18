"""Cloud management protocols and structured sync errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class CloudSyncError(Exception):
    """Raised when cloud synchronization or transfer operations fail."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"[{code}] {message} (retryable={retryable})")
        self.code = code
        self.message = message
        self.retryable = retryable


@runtime_checkable
class CloudManager(Protocol):
    """Base protocol for cloud storage and synchronization providers."""

    def sync(self, model_name: str, epoch: int, src_dir: Path, **kwargs: Any) -> bool:
        """Synchronize local artifacts to remote cloud target."""
        ...

    def pull(self, remote_identifier: str, dest_dir: Path, **kwargs: Any) -> bool:
        """Download remote artifacts to local destination."""
        ...

    def probe_health(self) -> dict[str, Any]:
        """Check cloud provider connectivity and credentials health."""
        ...
