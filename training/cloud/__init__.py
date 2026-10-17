"""Unified cloud management subsystem."""

from __future__ import annotations

from training.cloud.credentials import (
    find_kaggle_users_file,
    load_kaggle_users,
    mask_secret,
    resolve_gdrive_credentials,
    resolve_github_credentials,
    resolve_kaggle_credentials,
)
from training.cloud.gdrive import GDriveSyncManager, compute_file_sha256
from training.cloud.git_hub import GitHubSyncManager, git_hub_sync
from training.cloud.kaggle_hub import KaggleHubManager
from training.cloud.kaggle_model_sync import KaggleModelSyncManager, ModelPushResult, VariationPushResult
from training.cloud.manager import CloudManager, CloudSyncError
from training.cloud.metadata_generator import (
    KaggleMetadataService,
    generate_kaggle_metadata,
)

__all__ = [
    "CloudManager",
    "CloudSyncError",
    "GDriveSyncManager",
    "GitHubSyncManager",
    "KaggleHubManager",
    "KaggleMetadataService",
    "KaggleModelSyncManager",
    "ModelPushResult",
    "VariationPushResult",
    "compute_file_sha256",
    "find_kaggle_users_file",
    "generate_kaggle_metadata",
    "git_hub_sync",
    "load_kaggle_users",
    "mask_secret",
    "resolve_gdrive_credentials",
    "resolve_github_credentials",
    "resolve_kaggle_credentials",
]
