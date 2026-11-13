"""The PyTorch build the cloud training notebooks install on a CUDA GPU (Kaggle and Colab T4).

The requirement manifest the dependency cell installs (the env-manager's
``requirements-training.txt``) pins the torch family for the owner's host, whose wheel index
is newer than a T4 notebook's. Its ``torch==2.15.0`` has no ``cu126`` wheel, so with the
``cu126`` index only as an extra index pip took the CUDA 13 build from PyPI. The dependency
cell therefore installs :data:`CLOUD_CUDA_BUILD` from its own index first, drops the torch
family from the manifest and constrains the rest of the install to these exact builds.

This module is the one place that names the cloud torch build; the deps cell and the
hardware sentinel read it.
"""

from __future__ import annotations

from dataclasses import dataclass

TORCH_FAMILY: tuple[str, ...] = ("torch", "torchvision", "torchaudio")
"""Packages whose wheels are built per CUDA version and must come from one index."""


@dataclass(frozen=True)
class TorchBuild:
    """A torch family build: the wheel index and the exact pins installed from it."""

    accelerator: str
    """Accelerator label the dependency cell prints (``cuda_cu126``)."""

    index_url: str
    """PyTorch wheel index the pins are installed from (``--index-url``)."""

    pins: tuple[str, ...]
    """Exact ``name==version+local`` requirements, one per :data:`TORCH_FAMILY` package."""


CLOUD_CUDA_BUILD = TorchBuild(
    accelerator="cuda_cu126",
    index_url="https://download.pytorch.org/whl/cu126",
    pins=("torch==2.14.1+cu126", "torchvision==0.29.1+cu126", "torchaudio==2.11.0+cu126"),
)
"""CUDA 12.6 build for the Tesla T4 (sm_75) of Kaggle GPU T4 x2 and Colab. ``2.14.1`` is
the newest torch on the ``cu126`` index (``torchvision 0.29.1``, ``torchaudio 2.11.0``)."""
