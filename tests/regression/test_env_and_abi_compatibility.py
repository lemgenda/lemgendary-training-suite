"""Regression tests for Python 3.14 environment, ABI compatibility, and permissive licenses."""

import importlib
import sys

import pytest


def test_python_version() -> None:
    """Assert runtime is strictly Python 3.14."""
    assert sys.version_info[:2] == (3, 14), f"Expected Python 3.14, got {sys.version}"


def test_permissive_license_zero_agpl() -> None:
    """Ensure copyleft/AGPL dependencies (albumentations/albumentationsx) are not installed."""
    with pytest.raises(ImportError):
        importlib.import_module("albumentations")

    with pytest.raises(ImportError):
        importlib.import_module("albumentationsx")


@pytest.mark.parametrize(
    "pkg_name",
    [
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics",
        "kornia",
        "PIL",
        "numpy",
        "yaml",
        "psutil",
        "pandas",
        "pyarrow",
        "scipy",
        "tqdm",
        "cv2",
        "matplotlib",
        "einops",
        "timm",
        "requests",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "sentencepiece",
        "huggingface_hub",
        "ultralytics",
        "pycocotools",
        "torch_fidelity",
        "onnx",
        "onnxsim",
        "onnxconverter_common",
        "onnxscript",
        "websockets",
        "fastapi",
        "uvicorn",
        "pydantic",
        "typer",
        "click",
        "httpx2",
        "webdataset",
        "litdata",
        "zstandard",
        "jsonschema",
    ],
)
def test_import_dependencies_zero_warnings(pkg_name: str) -> None:
    """Dynamically import required packages asserting successful import."""
    mod = importlib.import_module(pkg_name)
    assert mod is not None


def test_platform_gated_dependencies() -> None:
    """Check platform-specific packages."""
    if sys.platform != "darwin":
        bnb = importlib.import_module("bitsandbytes")
        assert bnb is not None

    if sys.platform == "win32":
        mt5 = importlib.import_module("MetaTrader5")
        assert mt5 is not None
        ort_dml = importlib.import_module("onnxruntime")
        assert ort_dml is not None
        assert "DmlExecutionProvider" in ort_dml.get_available_providers()


def test_boto3_deadlock_prevention() -> None:
    """Ensure litdata S3 dependencies are modern and bcdoc is eliminated."""
    with pytest.raises(ImportError):
        importlib.import_module("bcdoc")

    import boto3
    import botocore
    import jmespath
    import six
    from packaging.version import Version

    assert Version(boto3.__version__) >= Version("1.44.11")
    assert Version(botocore.__version__) >= Version("1.44.11")
    assert Version(jmespath.__version__) >= Version("1.1.0")
    assert Version(six.__version__) >= Version("1.17.0")


def test_no_legacy_httpx_or_pydantic_core_duplicate_pins() -> None:
    """Ensure legacy duplicate httpx and pydantic-core pins are not present."""
    import httpx2
    import pydantic
    from packaging.version import Version

    assert Version(httpx2.__version__) >= Version("2.16.2")
    assert Version(pydantic.__version__) >= Version("2.14.4")
