"""Model registry and metadata resolver for notebook generation.

Extracts model specifications, input dimensions, repository handles,
and dataset manifold metadata into a structured ModelNotebookMeta record.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from training.ecosystem import kaggle_dataset_handle, kaggle_owner
from training.registry.kaggle_datasets import model_kaggle_dataset_handles
from training.utils.lazy_imports import load_cloud_module

logger = logging.getLogger("lemtrain.notebooks.registry")


def _get_project_root() -> Path:
    """Resolve project root directory across training suite or datasets repositories."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (
            (parent / "config.yaml").exists()
            or (parent / "unified_data.yaml").exists()
            or (parent / "pyproject.toml").exists()
        ):
            return parent
    return here.parent.parent.parent


@dataclass(frozen=True)
class NotebookIdentity:
    """Model key, its derived names and family."""

    model_key: str
    pascal_name: str
    kebab_name: str
    filename: str
    is_forex: bool


@dataclass(frozen=True)
class NotebookDataset:
    """Dataset manifold the notebook mounts or downloads."""

    dataset_slug: str
    ds_list: list[str]
    ds_keys_repr: str
    clean_kaggle_repo: str
    primary_manifold: str
    no_download: bool


@dataclass(frozen=True)
class KaggleNotebookHandles:
    """Kaggle owner, model slug and per-variation model handles."""

    username: str
    slug: str
    handles: dict[str, str] = field(default_factory=dict)
    pytorch_default: str = ""  # Primary handle, used to pull checkpoints
    pytorch_fp32: str = ""
    onnx_fp16: str = ""
    onnx_fp32: str = ""


@dataclass(frozen=True)
class ModelNotebookMeta:
    """Immutable model metadata record consumed by notebook cell generators."""

    identity: NotebookIdentity
    dataset: NotebookDataset
    kaggle: KaggleNotebookHandles
    input_size: tuple[int, int]
    preferred_parallel: str = ""


def model_dataset_list(model_info: dict[str, Any]) -> list[str]:
    """Return a registry entry's ``datasets`` (or legacy ``dataset``) value as a list."""
    ds_raw = model_info.get("datasets", []) or model_info.get("dataset", [])
    if isinstance(ds_raw, str):
        return [ds_raw]
    if isinstance(ds_raw, (list, tuple)):
        return list(ds_raw)
    return []


def _dataset_slug(config: dict[str, Any] | None, pascal_name: str, default: str) -> str:
    """Kaggle dataset slug from ``kaggle_dataset_urls`` (dict by name, or first list entry)."""
    if not config:
        return default
    k_urls = config.get("kaggle_dataset_urls", {})
    if isinstance(k_urls, dict):
        for key, url in k_urls.items():
            if pascal_name in key:
                return url.split("/")[-1]
    elif isinstance(k_urls, list) and k_urls:
        return k_urls[0].split("/")[-1]
    return default


def _kaggle_username(model_key: str) -> str:
    """Kaggle username from credentials, else the configured owner."""
    try:
        credentials = load_cloud_module("credentials")
        user_cred, _ = credentials.resolve_kaggle_credentials()
        return user_cred or kaggle_owner()
    except (ImportError, OSError, ValueError) as exc:
        logger.warning(
            "Kaggle credential resolution failed for %s; using configured username: %s",
            model_key,
            exc,
        )
        return kaggle_owner()


def _variation_handles(k_username: str, k_slug: str) -> dict[str, str]:
    """Kaggle Models handle per variation key."""
    variations = load_cloud_module("variations")
    return {
        variation.key: variation.handle(k_username, k_slug)
        for variation in (
            variations.PYTORCH_DEFAULT,
            variations.PYTORCH_FP32,
            variations.ONNX_FP16,
            variations.ONNX_FP32,
        )
    }


def _notebook_input_size(size_raw: Any) -> tuple[int, int]:
    """``(H, W)`` from ``input_size`` (``[C, H, W]``, ``[H, W]`` or int); 256x256 otherwise."""
    if isinstance(size_raw, list):
        if len(size_raw) == 3:
            return int(size_raw[1]), int(size_raw[2])
        if len(size_raw) >= 2:
            return int(size_raw[0]), int(size_raw[1])
        return 256, 256
    if isinstance(size_raw, int):
        return size_raw, size_raw
    return 256, 256


def resolve_model_metadata(
    model_key: str,
    registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ModelNotebookMeta:
    """Resolve comprehensive notebook generation metadata for a given model key."""
    pascal_name = model_key.replace("_", " ").title().replace(" ", "")
    kebab_name = model_key.replace("_", "-")

    model_info: dict[str, Any] = registry.get(model_key, {}) if registry else {}
    ds_list = model_dataset_list(model_info)

    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()

    # Resolve canonical Kaggle username and handles using SSOT variations
    k_username = _kaggle_username(model_key)
    k_slug = (
        model_info.get("kaggle_metadata", {}).get("slug")
        if isinstance(model_info.get("kaggle_metadata"), dict)
        else None
    )
    if not k_slug:
        k_slug = model_key.replace("_", "-").lower()
    handles = _variation_handles(k_username, k_slug)
    variations = load_cloud_module("variations")

    dataset_handles = model_kaggle_dataset_handles(model_info)
    primary_manifold = (
        ds_list[0]
        if ds_list
        else (f"LemGendized{pascal_name}" if not is_forex else "LemGendizedForexUniverse")
    )
    clean_kaggle_repo = (
        dataset_handles[0] if dataset_handles else kaggle_dataset_handle(primary_manifold)
    )

    return ModelNotebookMeta(
        identity=NotebookIdentity(
            model_key=model_key,
            pascal_name=pascal_name,
            kebab_name=kebab_name,
            filename=model_info.get("filename", model_key),
            is_forex=is_forex,
        ),
        dataset=NotebookDataset(
            dataset_slug=_dataset_slug(config, pascal_name, f"lemgendary-{kebab_name}"),
            ds_list=ds_list,
            ds_keys_repr=repr(
                [model_key.lower(), model_key.replace("_", "-"), model_key.replace("_", "")]
                + [d.lower() for d in ds_list]
                + (["forex"] if is_forex else [])
            ),
            clean_kaggle_repo=clean_kaggle_repo,
            primary_manifold=primary_manifold,
            no_download=bool(config.get("notebook_no_download", False)) if config else False,
        ),
        kaggle=KaggleNotebookHandles(
            username=k_username,
            slug=k_slug,
            handles=handles,
            pytorch_default=handles[variations.PYTORCH_DEFAULT.key],
            pytorch_fp32=handles[variations.PYTORCH_FP32.key],
            onnx_fp16=handles[variations.ONNX_FP16.key],
            onnx_fp32=handles[variations.ONNX_FP32.key],
        ),
        input_size=_notebook_input_size(model_info.get("input_size", [3, 256, 256])),
        preferred_parallel=str(model_info.get("preferred_parallel", "") or ""),
    )


def load_registry(
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load neural network registry manifest (unified_models_v2.yaml)."""
    root = project_root if project_root is not None else _get_project_root()

    reg_name = "unified_models_v2.yaml"
    if config and "unified_models" in config:
        reg_name = str(config["unified_models"])

    reg_path = root / reg_name
    if not reg_path.exists():
        return {}

    with open(reg_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content if isinstance(content, dict) else {}
