"""Delegation adapter for LemGendary Dataset Compiler Suite."""

import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
import yaml

from training.utils.paths import get_workspace_root

logger = logging.getLogger("lemtrain.dataset_delegate")


class DatasetCompilerDelegate:
    """Provides typed resolution of compiled manifolds and metadata from LemGendary Dataset Compiler Suite."""

    def __init__(self, workspace_root: Path | None = None, sidecar_port: int = 8100) -> None:
        self.workspace_root = workspace_root or get_workspace_root()
        self.datasets_dir = self.workspace_root / "lemgendary-datasets"
        self.manifolds_repo = self.workspace_root / "LemGendaryDatasets"
        self.sidecar_url = f"http://127.0.0.1:{sidecar_port}"

    def is_sidecar_online(self) -> bool:
        """Probe dataset compiler sidecar API health."""
        try:
            req = urllib.request.Request(f"{self.sidecar_url}/api/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

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
                except Exception as exc:
                    logger.warning("Failed parsing models metadata at %s: %s", candidate, exc)
        return {}

    def resolve_manifold(self, manifold_name: str) -> Path | None:
        """Resolve directory path for a compiled manifold."""
        # 1. Check LemGendaryDatasets repository
        target = self.manifolds_repo / manifold_name
        if target.exists() and target.is_dir():
            return target

        # 2. Check local data/datasets cache inside training suite
        local_target = self.workspace_root / "lemgendary-training-suite" / "data" / "datasets" / manifold_name
        if local_target.exists() and local_target.is_dir():
            return local_target

        # 3. Check within datasets project directory
        alt_target = self.datasets_dir / "manifolds" / manifold_name
        if alt_target.exists() and alt_target.is_dir():
            return alt_target

        return None
