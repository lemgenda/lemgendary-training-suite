"""Dataset discovery and manifold inspection endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import APIRouter
import yaml

from training.utils.paths import get_project_root

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> list[dict[str, Any]]:
    """List datasets defined across the repository and manifests."""
    root = get_project_root()
    datasets_list: list[dict[str, Any]] = []

    # 1. Inspect config.yaml and unified_models_v2.yaml
    cfg_path = root / "config.yaml"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    reg_name = cfg.get("unified_models", "unified_models_v2.yaml")
    reg_path = root / reg_name
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            meta_datasets = data.get("_registry_metadata", {}).get("datasets", {})
            for name, meta in meta_datasets.items():
                datasets_list.append({
                    "name": name,
                    "count": meta.get("count"),
                    "type": meta.get("type", "compiled_manifold"),
                })

    # 2. Check local data directory
    data_dir = root / "data" / "datasets"
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and not any(x["name"] == d.name for x in datasets_list):
                datasets_list.append({
                    "name": d.name,
                    "path": str(d),
                    "type": "local_directory",
                })

    return datasets_list
