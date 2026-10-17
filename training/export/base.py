"""Base contracts and interfaces for model exporters in LemGendary Training Suite.

Establishes S.O.L.I.D. architectural foundation (Interface Segregation and Open-Closed principles)
for multi-framework and multi-precision model export.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from training.registry import ModelSpecification, get_model_registry
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.base")


class BaseModelExporter(ABC):
    """Abstract interface for all model exporters."""

    @abstractmethod
    def export(
        self,
        model: Any,
        model_key: str,
        output_dir: Path | str | None = None,
        spec: ModelSpecification | None = None,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Export model into framework variations and return mapped output paths.

        Returns a dictionary mapping format keys (e.g. 'pt', 'onnx_fp32', 'onnx_fp32_data', 'onnx_fp16')
        to their respective Path locations on disk.
        """

    @abstractmethod
    def supports(self, model_key: str, spec: ModelSpecification | None = None) -> bool:
        """Determine whether this exporter handles the specified model key and specification."""

    def resolve_spec(self, model_key: str, spec: ModelSpecification | None = None) -> ModelSpecification | None:
        """Helper to resolve ModelSpecification from registry if not provided directly."""
        if spec is not None:
            return spec
        reg = get_model_registry()
        return reg.get_model(model_key)

    def resolve_destination(self, model_key: str, output_dir: Path | str | None = None) -> Path:
        """Resolve the authoritative output directory in LemGendaryModels."""
        if output_dir:
            dest = Path(output_dir).resolve()
        else:
            root = get_project_root()
            dest = (root.parent / "LemGendaryModels" / model_key).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        return dest
