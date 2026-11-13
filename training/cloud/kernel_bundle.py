"""The one Kaggle kernel bundle builder (K-NB-1, K-GUI-1).

A bundle is a folder with the generated training notebook and ``kernel-metadata.json``,
ready for ``KaggleApi.kernels_push``. The GUI "Train on Kaggle" job, the ``lemtrain kaggle``
commands and the interactive launcher all build it here, and the notebook comes from the
same generator as the notebook files (:func:`training.notebooks.builders.kaggle.
build_training_notebook`), so a GUI-launched kernel runs exactly the notebook a user would
upload by hand.

Metadata (kaggle 2.x ``kernels_push`` keys):

* ``id`` ``<kernel owner>/<model slug>-training``; ``title`` slugifies to the same slug;
* ``code_file`` the notebook, ``kernel_type`` ``notebook``, ``language`` ``python``;
* ``is_private`` true, ``enable_internet`` true (the notebook clones the suite);
* ``enable_gpu`` / ``enable_tpu`` and ``machine_shape`` from the ecosystem accelerator table
  (:func:`training.ecosystem.kaggle_accelerator`); retired shapes are rejected;
* ``dataset_sources`` the model's manifold dataset handles from the registry;
* ``model_sources`` the model's ``pytorch/default`` Hub instance at its latest version, when
  the Hub has one (resume input; the recovery cell compares stored epochs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.ecosystem import (
    EcosystemFileError,
    KaggleAccelerator,
    kaggle_accelerator,
    kaggle_retired_machine_shapes,
)
from training.notebooks.builders.base import write_notebook
from training.notebooks.builders.kaggle import build_training_notebook
from training.notebooks.launch_options import LaunchOptions
from training.notebooks.registry import ModelNotebookMeta, load_registry, resolve_model_metadata
from training.registry.kaggle_datasets import model_kaggle_dataset_handles

METADATA_FILE = "kernel-metadata.json"
"""File name ``KaggleApi.kernels_push`` reads the kernel metadata from."""
KERNEL_SLUG_SUFFIX = "-training"
MAX_KERNEL_SLUG_LENGTH = 50
"""Kaggle rejects kernel titles (and so slugs) longer than 50 characters."""


class KernelBundleError(ValueError):
    """The bundle cannot be built (unknown model, unknown or retired accelerator)."""


def kernel_slug(model_slug: str) -> str:
    """Kernel slug of a model: its registry Kaggle slug plus ``-training``."""
    slug = f"{model_slug}{KERNEL_SLUG_SUFFIX}"
    if len(slug) > MAX_KERNEL_SLUG_LENGTH:
        raise KernelBundleError(f"Kernel slug '{slug}' exceeds {MAX_KERNEL_SLUG_LENGTH} characters")
    return slug


def kernel_title(model_slug: str) -> str:
    """Kernel title whose Kaggle slug is :func:`kernel_slug` (words of the slug, title-cased)."""
    return " ".join(part.capitalize() for part in kernel_slug(model_slug).split("-"))


def kernel_handle(kernel_owner: str, model_slug: str) -> str:
    """``<kernel owner>/<kernel slug>``: the ``id`` of the bundle and the status handle."""
    return f"{kernel_owner}/{kernel_slug(model_slug)}"


def checked_accelerator(accelerator_id: str | None) -> KaggleAccelerator:
    """The ecosystem accelerator ``accelerator_id`` (default when None), never a retired shape."""
    try:
        accelerator = kaggle_accelerator(accelerator_id)
    except EcosystemFileError as exc:
        raise KernelBundleError(str(exc)) from exc
    retired = {shape.lower() for shape in kaggle_retired_machine_shapes()}
    if accelerator.machine_shape and accelerator.machine_shape.lower() in retired:
        raise KernelBundleError(
            f"Kaggle accelerator '{accelerator.id}' uses the retired machine shape "
            f"'{accelerator.machine_shape}'"
        )
    return accelerator


@dataclass(frozen=True)
class KernelSources:
    """Inputs attached to the kernel session."""

    datasets: tuple[str, ...]
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelBundleSpec:
    """Everything ``kernel-metadata.json`` is built from."""

    model_key: str
    model_slug: str
    kernel_owner: str
    accelerator: KaggleAccelerator
    sources: KernelSources

    @property
    def handle(self) -> str:
        """``<kernel owner>/<kernel slug>``."""
        return kernel_handle(self.kernel_owner, self.model_slug)

    @property
    def notebook_file(self) -> str:
        """File name of the generated notebook inside the bundle."""
        return f"{self.model_key}_kaggle_training.ipynb"


def kernel_metadata(spec: KernelBundleSpec) -> dict[str, Any]:
    """``kernel-metadata.json`` content of ``spec``."""
    metadata: dict[str, Any] = {
        "id": spec.handle,
        "title": kernel_title(spec.model_slug),
        "code_file": spec.notebook_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": spec.accelerator.kind == "gpu",
        "enable_tpu": spec.accelerator.kind == "tpu",
        "enable_internet": True,
        "dataset_sources": list(spec.sources.datasets),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": list(spec.sources.models),
    }
    if spec.accelerator.machine_shape:
        metadata["machine_shape"] = spec.accelerator.machine_shape
    return metadata


@dataclass(frozen=True)
class KernelBundle:
    """A written bundle folder."""

    directory: Path
    spec: KernelBundleSpec
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def handle(self) -> str:
        """``<kernel owner>/<kernel slug>``."""
        return self.spec.handle


@dataclass(frozen=True)
class BundleRequest:
    """What to build: the model, who pushes it, on which accelerator, resuming which version."""

    model_key: str
    kernel_owner: str
    accelerator_id: str | None = None
    resume_version: int | None = None
    """Latest ``pytorch/default`` Hub version to mount (None: the Hub has none)."""
    override_frozen: bool = False
    """Trainer ``--override-frozen``: resume although a frozen run setting changed."""


def _registry_entry(model_key: str, registry: dict[str, Any]) -> dict[str, Any]:
    entry = registry.get(model_key)
    if not isinstance(entry, dict) or model_key.startswith("_"):
        raise KernelBundleError(f"Model '{model_key}' not found in the registry")
    return entry


def bundle_spec(
    request: BundleRequest, registry: dict[str, Any] | None = None
) -> tuple[KernelBundleSpec, ModelNotebookMeta]:
    """Resolve the bundle spec and the notebook metadata of ``request``."""
    reg = registry if registry is not None else load_registry()
    entry = _registry_entry(request.model_key, reg)
    meta = resolve_model_metadata(request.model_key, reg)
    models: tuple[str, ...] = ()
    if request.resume_version is not None:
        models = (f"{meta.kaggle.pytorch_default}/{int(request.resume_version)}",)
    spec = KernelBundleSpec(
        model_key=request.model_key,
        model_slug=meta.kaggle.slug,
        kernel_owner=request.kernel_owner,
        accelerator=checked_accelerator(request.accelerator_id),
        sources=KernelSources(datasets=tuple(model_kaggle_dataset_handles(entry)), models=models),
    )
    return spec, meta


def build_kernel_bundle(
    request: BundleRequest, out_dir: Path, registry: dict[str, Any] | None = None
) -> KernelBundle:
    """Write the notebook and ``kernel-metadata.json`` of ``request`` into ``out_dir``.

    Nothing else is written: no manifold or workspace copies, no credential files.
    """
    spec, meta = bundle_spec(request, registry)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = LaunchOptions(override_frozen=request.override_frozen)
    notebook = build_training_notebook(meta, options)
    if write_notebook(notebook, out_dir / spec.notebook_file) is None:
        raise KernelBundleError(f"Could not write the training notebook of {spec.model_key}")
    metadata = kernel_metadata(spec)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return KernelBundle(directory=out_dir, spec=spec, metadata=metadata)


def model_kernel_handle(
    kernel_owner: str, model_key: str, registry: dict[str, Any] | None = None
) -> str:
    """``<kernel owner>/<kernel slug>`` of a registry model (what the bundle pushes)."""
    reg = registry if registry is not None else load_registry()
    _registry_entry(model_key, reg)
    return kernel_handle(kernel_owner, resolve_model_metadata(model_key, reg).kaggle.slug)
