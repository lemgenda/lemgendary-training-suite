"""Delegation adapter for LemGendary Dataset Compiler Suite."""

import logging
from pathlib import Path
from typing import Any

import yaml

from training import ecosystem
from training.utils.paths import get_workspace_root
from training.utils.sidecar import sidecar_online

logger = logging.getLogger("lemtrain.dataset_delegate")


class DatasetCompilerDelegate:
    """Typed resolution of compiled manifolds and metadata from the LemGendary Dataset Compiler."""

    def __init__(self, workspace_root: Path | None = None, sidecar_url: str | None = None) -> None:
        self.workspace_root = workspace_root or get_workspace_root()
        datasets_service = ecosystem.ecosystem().service(ecosystem.DATASETS_SERVICE)
        self.datasets_dir = self.workspace_root / datasets_service.project
        self.manifolds_repo = self.workspace_root / "LemGendaryDatasets"
        self.sidecar_url = sidecar_url or ecosystem.service_url(ecosystem.DATASETS_SERVICE)

    def is_sidecar_online(self) -> bool:
        """Probe dataset compiler sidecar API health."""
        return sidecar_online(self.sidecar_url)

    def resolve_models_metadata(self) -> dict[str, Any]:
        """Resolve canonical models_metadata.yaml from lemgendary-datasets/models/ with fallback."""
        candidates = [
            self.datasets_dir / "models" / "models_metadata.yaml",
            self.datasets_dir / "models_metadata.yaml",
            self.workspace_root / "lemgendary-training-suite" / "models" / "models_metadata.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    content = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                    if isinstance(content, dict):
                        return content
                except (OSError, yaml.YAMLError) as exc:
                    logger.warning("Failed parsing models metadata at %s: %s", candidate, exc)
        return {}

    def resolve_manifold(self, manifold_name: str) -> Path | None:
        """Resolve directory path for a compiled manifold."""
        # 1. Check LemGendaryDatasets repository
        target = self.manifolds_repo / manifold_name
        if target.exists() and target.is_dir():
            return target

        # 2. Check local data/datasets cache inside training suite
        local_target = (
            self.workspace_root / "lemgendary-training-suite" / "data" / "datasets" / manifold_name
        )
        if local_target.exists() and local_target.is_dir():
            return local_target

        # 3. Check within datasets project directory
        alt_target = self.datasets_dir / "manifolds" / manifold_name
        if alt_target.exists() and alt_target.is_dir():
            return alt_target

        return None
