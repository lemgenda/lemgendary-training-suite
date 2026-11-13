"""Unified cloud management subsystem.

The package exports are resolved lazily (PEP 562): importing a light submodule such as
:mod:`training.cloud.trainer_exit` or :mod:`training.cloud.session_deadline` must not pull in
the export stack (``models.factory`` and torchvision). The generated Kaggle notebook imports
``trainer_exit`` in its own kernel process, which loaded the preinstalled torch in the
hardware sentinel before the dependency cell installed a different one; importing
torchvision there fails (``operator torchvision::nms does not exist``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.cloud.credentials import (
        load_kaggle_users,
        mask_secret,
        resolve_gdrive_credentials,
        resolve_github_credentials,
        resolve_kaggle_credentials,
    )
    from training.cloud.gdrive import GDriveSyncManager, compute_file_sha256
    from training.cloud.git_hub import GitHubSyncManager, git_hub_sync
    from training.cloud.kaggle_api_sync import KaggleApiSyncService
    from training.cloud.kaggle_hub import KaggleHubManager
    from training.cloud.kaggle_model_sync import (
        KaggleModelSyncManager,
        ModelPushResult,
        VariationPushResult,
    )
    from training.cloud.manager import CloudManager, CloudSyncError
    from training.cloud.metadata_generator import (
        KaggleMetadataService,
        generate_kaggle_metadata,
        load_model_metadata,
    )
    from training.cloud.model_metadata import ModelMetadata, ModelSection, VariationSection
    from training.cloud.variations import (
        ONNX_FP16,
        ONNX_FP32,
        PYTORCH_DEFAULT,
        PYTORCH_FP32,
        VARIATIONS,
        KaggleVariation,
    )

_EXPORTS: dict[str, str] = {
    "load_kaggle_users": "credentials",
    "mask_secret": "credentials",
    "resolve_gdrive_credentials": "credentials",
    "resolve_github_credentials": "credentials",
    "resolve_kaggle_credentials": "credentials",
    "GDriveSyncManager": "gdrive",
    "compute_file_sha256": "gdrive",
    "GitHubSyncManager": "git_hub",
    "git_hub_sync": "git_hub",
    "KaggleApiSyncService": "kaggle_api_sync",
    "KaggleHubManager": "kaggle_hub",
    "KaggleModelSyncManager": "kaggle_model_sync",
    "ModelPushResult": "kaggle_model_sync",
    "VariationPushResult": "kaggle_model_sync",
    "CloudManager": "manager",
    "CloudSyncError": "manager",
    "KaggleMetadataService": "metadata_generator",
    "generate_kaggle_metadata": "metadata_generator",
    "load_model_metadata": "metadata_generator",
    "ModelMetadata": "model_metadata",
    "ModelSection": "model_metadata",
    "VariationSection": "model_metadata",
    "ONNX_FP16": "variations",
    "ONNX_FP32": "variations",
    "PYTORCH_DEFAULT": "variations",
    "PYTORCH_FP32": "variations",
    "VARIATIONS": "variations",
    "KaggleVariation": "variations",
}
"""Exported name -> submodule of :mod:`training.cloud` that defines it."""


def __getattr__(name: str) -> Any:
    """Import the submodule of an exported name on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Module attributes plus the lazily resolved exports."""
    return sorted({*globals(), *_EXPORTS})


__all__ = [
    "ONNX_FP16",
    "ONNX_FP32",
    "PYTORCH_DEFAULT",
    "PYTORCH_FP32",
    "VARIATIONS",
    "CloudManager",
    "CloudSyncError",
    "GDriveSyncManager",
    "GitHubSyncManager",
    "KaggleApiSyncService",
    "KaggleHubManager",
    "KaggleMetadataService",
    "KaggleModelSyncManager",
    "KaggleVariation",
    "ModelMetadata",
    "ModelPushResult",
    "ModelSection",
    "VariationPushResult",
    "VariationSection",
    "compute_file_sha256",
    "generate_kaggle_metadata",
    "git_hub_sync",
    "load_kaggle_users",
    "load_model_metadata",
    "mask_secret",
    "resolve_gdrive_credentials",
    "resolve_github_credentials",
    "resolve_kaggle_credentials",
]
