"""Loaders for heavy, optional or cycle-prone dependencies.

Single responsibility: every package that is too slow to import eagerly, is
optional in ``requirements.txt`` (absent from light installs and the test
environment) or would close an import cycle when imported at module level is
imported here and nowhere else.

Each loader imports through ``importlib.import_module`` when it is called, so
importing a caller never pulls in transformers, peft, diffusers, ultralytics,
the ONNX tool chain, the Kaggle SDKs or the torch-backed training modules. A
missing package raises ``ImportError`` (``ModuleNotFoundError``) from the
loader, which callers already handle.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

if TYPE_CHECKING:
    from torch.nn import Module

_T = TypeVar("_T")


class KaggleIamTypes(NamedTuple):
    """The kagglesdk modules needed to edit a model's IAM policy."""

    iam: ModuleType
    security: ModuleType


def is_installed(package: str) -> bool:
    """Return True when the top-level ``package`` can be imported."""
    return importlib.util.find_spec(package) is not None


def load_optional(loader: Callable[[], _T]) -> _T | None:
    """Return ``loader()``, or None when the package it imports is not installed."""
    try:
        return loader()
    except ImportError:
        return None


# --- Third-party optional packages ---------------------------------------------------


def load_transformers() -> ModuleType:
    """Return the ``transformers`` module (Hugging Face model classes)."""
    return importlib.import_module("transformers")


def load_torch_xla() -> ModuleType:
    """Return the ``torch_xla`` module (PyTorch/XLA for Cloud TPU)."""
    return importlib.import_module("torch_xla")


def load_torch_xla_model() -> ModuleType:
    """Return ``torch_xla.core.xla_model``."""
    return importlib.import_module("torch_xla.core.xla_model")


def load_peft() -> ModuleType:
    """Return the ``peft`` module (LoRA adapters)."""
    return importlib.import_module("peft")


def load_diffusers() -> ModuleType:
    """Return the ``diffusers`` module (SDXL and FLUX.1 components and schedulers)."""
    return importlib.import_module("diffusers")


def load_ultralytics() -> ModuleType:
    """Return the ``ultralytics`` module (YOLO models)."""
    return importlib.import_module("ultralytics")


def load_cv2() -> ModuleType:
    """Return the OpenCV ``cv2`` module.

    OpenCV 5 installs ``cv2`` as a Python loader package whose ``__init__`` replaces
    itself with the native extension at import time, so static analysers see none of
    its members. Callers use the module this loader returns, which is the native one.
    """
    return importlib.import_module("cv2")


def load_onnx() -> ModuleType:
    """Return the ``onnx`` module with its ``external_data_helper`` submodule loaded."""
    importlib.import_module("onnx.external_data_helper")
    return importlib.import_module("onnx")


def load_onnxsim() -> ModuleType:
    """Return the ``onnxsim`` module (ONNX graph simplifier)."""
    return importlib.import_module("onnxsim")


def load_onnx_float16() -> ModuleType:
    """Return ``onnxconverter_common.float16`` (FP32 -> FP16 ONNX conversion)."""
    return importlib.import_module("onnxconverter_common.float16")


def load_onnxruntime() -> ModuleType:
    """Return the ``onnxruntime`` module (inference sessions)."""
    return importlib.import_module("onnxruntime")


def load_mean_average_precision() -> Any:
    """Return torchmetrics' ``MeanAveragePrecision`` metric class."""
    module = importlib.import_module("torchmetrics.detection.mean_ap")
    return module.MeanAveragePrecision


def load_lpips_metric() -> type[Module]:
    """Return torchmetrics' ``LearnedPerceptualImagePatchSimilarity`` metric class."""
    module = importlib.import_module("torchmetrics.image.lpip")
    return module.LearnedPerceptualImagePatchSimilarity


def load_fid_metric() -> type[Module]:
    """Return torchmetrics' ``FrechetInceptionDistance`` metric class."""
    module = importlib.import_module("torchmetrics.image.fid")
    return module.FrechetInceptionDistance


def load_streaming() -> ModuleType:
    """Return the ``streaming`` (mosaicml-streaming) module."""
    return importlib.import_module("streaming")


def load_kagglehub() -> ModuleType:
    """Return the ``kagglehub`` module."""
    return importlib.import_module("kagglehub")


def load_kaggle_api() -> Any:
    """Return the ``KaggleApi`` class.

    ``import kaggle`` authenticates on import and raises ``OSError`` when no
    credentials exist, so it is deferred until a Kaggle call is made.
    """
    module = importlib.import_module("kaggle.api.kaggle_api_extended")
    return module.KaggleApi


def load_kaggle_iam() -> KaggleIamTypes:
    """Return the kagglesdk IAM request types and the security enums they use."""
    return KaggleIamTypes(
        iam=importlib.import_module("kagglesdk.security.types.iam_service"),
        security=importlib.import_module("kagglesdk.security.types.security_types"),
    )


def load_pandas_ta() -> ModuleType:
    """Return the ``pandas_ta_classic`` technical-indicator module."""
    return importlib.import_module("pandas_ta_classic")


def load_torch_directml() -> ModuleType:
    """Return the ``torch_directml`` backend module (DirectML devices are ``privateuseone``)."""
    return importlib.import_module("torch_directml")


# --- Suite modules that are heavy or would close an import cycle -----------------------


def load_model_module(name: str) -> ModuleType:
    """Return ``models.<name>``; the heavy model modules import diffusers or transformers."""
    return importlib.import_module(f"models.{name}")


def load_variation_staging() -> Any:
    """Return the ``VariationStagingService`` class.

    ``training.export`` imports torch, and ``training.export.staging`` imports
    ``training.cloud``, so a module-level import from ``training.cloud`` is circular.
    """
    module = importlib.import_module("training.export.staging")
    return module.VariationStagingService


def load_pareto_soup() -> Any:
    """Return ``training.export.soup.create_pareto_model_soup`` (imports torch)."""
    module = importlib.import_module("training.export.soup")
    return module.create_pareto_model_soup


def load_cloud_module(name: str) -> ModuleType:
    """Return ``training.cloud.<name>``.

    The ``training.cloud`` package imports torch and the Kaggle SDK, while
    notebook and README generation stay light.
    """
    return importlib.import_module(f"training.cloud.{name}")


def load_hub_readme_generator() -> Any:
    """Return ``generate_hub_readme``; the hub generator imports training.telemetry (torch)."""
    module = importlib.import_module("training.hub_readme_generator")
    return module.generate_hub_readme
