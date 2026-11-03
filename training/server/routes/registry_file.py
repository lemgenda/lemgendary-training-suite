"""Location of the unified model registry YAML for the server routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def registry_path(project_root: Path) -> Path:
    """Return the registry YAML named by ``config.yaml`` (default ``unified_models_v2.yaml``)."""
    cfg_path = project_root / "config.yaml"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return project_root / cfg.get("unified_models", "unified_models_v2.yaml")
