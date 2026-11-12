"""Build a kernel bundle in a temporary folder and push it with the Kaggle API.

Shared by the GUI "Train on Kaggle" job, ``lemtrain kaggle launch``, the legacy
``kaggle_cloud_manager`` entry point and the interactive launcher. The bundle never lands
in the git working tree (K-GUI-4) and there is no CLI subprocess fallback: the push uses
the authenticated ``KaggleApi`` the caller passes in.
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.cloud.hub_versions import VersionLookup, kaggle_version_lookup
from training.cloud.kernel_bundle import (
    BundleRequest,
    KernelBundle,
    build_kernel_bundle,
    bundle_spec,
)
from training.notebooks.registry import load_registry

logger = logging.getLogger("lemtrain.cloud.kernel_launch")


class KernelLaunchError(RuntimeError):
    """Kaggle refused the kernel push."""


@dataclass(frozen=True)
class KernelLaunch:
    """A pushed kernel: its handle, the metadata it was pushed with and Kaggle's answer."""

    handle: str
    metadata: dict[str, Any]
    version_number: int | None
    url: str


def _push(api: Any, bundle: KernelBundle) -> KernelLaunch:
    response = api.kernels_push(str(bundle.directory))
    error = str(getattr(response, "error", "") or "")
    invalid = [
        f"{name}: {value}"
        for name in (
            "invalid_dataset_sources",
            "invalid_model_sources",
            "invalid_kernel_sources",
            "invalid_tags",
        )
        if (value := getattr(response, name, None))
    ]
    if error or invalid:
        raise KernelLaunchError(
            f"Kaggle refused kernel {bundle.handle}: {'; '.join([error, *invalid]).strip('; ')}"
        )
    version = getattr(response, "version_number", None)
    return KernelLaunch(
        handle=bundle.handle,
        metadata=bundle.metadata,
        version_number=version if isinstance(version, int) else None,
        url=str(getattr(response, "url", "") or ""),
    )


def launch_kernel(
    api: Any,
    request: BundleRequest,
    versions: VersionLookup | None = None,
    registry: dict[str, Any] | None = None,
) -> KernelLaunch:
    """Build the bundle of ``request`` (resuming the latest Hub version) and push it.

    ``request.resume_version`` is filled from ``versions`` (default: the Kaggle API) when it
    is None; a model without a ``pytorch/default`` Hub version starts without
    ``model_sources``.
    """
    reg = registry if registry is not None else load_registry()
    if request.resume_version is None:
        _, meta = bundle_spec(request, reg)
        lookup = versions if versions is not None else kaggle_version_lookup(api)
        request = dataclasses.replace(request, resume_version=lookup(meta.kaggle.pytorch_default))
    with tempfile.TemporaryDirectory(prefix=f"lemtrain-kernel-{request.model_key}-") as tmp:
        bundle = build_kernel_bundle(request, Path(tmp), reg)
        logger.info(
            "Pushing kernel %s (%s, resume version %s)",
            bundle.handle,
            bundle.spec.accelerator.id,
            request.resume_version,
        )
        return _push(api, bundle)
