"""Structured machine-readable reports for cloud sync operations.

Provides standardized serialization for push and pull synchronization reports
compatible with CLI flags (--report <path>, --json) and pipeline automation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VariationSyncReport:
    """Detailed sync result for a single model variation."""

    variation: str
    handle: str
    success: bool
    status: str
    files: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)
    checksum_verified: bool = False
    idempotent_noop: bool = False
    retries: int = 0
    version: int | None = None
    error: str | None = None


@dataclass
class ModelSyncReport:
    """Machine-readable report summarizing a model push or pull operation."""

    action: str
    model_key: str
    success: bool
    timestamp_utc: str
    elapsed_seconds: float
    variations: dict[str, VariationSyncReport] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_to(self, path: Path | str) -> None:
        """Write report to destination file path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
