"""Exporter Factory for LemGendary Training Suite.

Follows the Factory Method pattern to resolve the appropriate BaseModelExporter
implementation based on architecture type, category, and model key.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from training.export.base import BaseModelExporter
from training.export.forex_exporter import ForexModelExporter
from training.export.llm_exporter import LLMModelExporter
from training.export.pytorch_exporter import PyTorchModelExporter
from training.export.yolo_exporter import YOLOModelExporter
from training.registry import ModelSpecification, get_model_registry

logger = logging.getLogger("lemtrain.export.factory")


class ExporterFactory:
    """Factory creating and resolving concrete model exporters."""

    _exporters: ClassVar[list[BaseModelExporter]] = [
        LLMModelExporter(),
        YOLOModelExporter(),
        ForexModelExporter(),
        PyTorchModelExporter(),  # Default / general PyTorch exporter
    ]

    @classmethod
    def get_exporter(
        cls,
        model_key: str,
        spec: ModelSpecification | None = None,
    ) -> BaseModelExporter:
        """Resolve the appropriate exporter for a given model."""
        resolved_spec = spec
        if resolved_spec is None:
            reg = get_model_registry()
            resolved_spec = reg.get_model(model_key)

        for exporter in cls._exporters:
            if exporter.supports(model_key, resolved_spec):
                return exporter

        # Fallback to PyTorchModelExporter
        return cls._exporters[-1]


def get_exporter_for_model(model_key: str, spec: ModelSpecification | None = None) -> BaseModelExporter:
    """Convenience accessor to obtain an exporter for a model."""
    return ExporterFactory.get_exporter(model_key, spec)
