"""Unified cloud management subsystem."""

from __future__ import annotations

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
