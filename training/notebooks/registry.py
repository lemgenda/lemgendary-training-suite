"""Model registry and metadata resolver for notebook generation.

Extracts model specifications, input dimensions, repository handles,
and dataset manifold metadata into a structured ModelNotebookMeta record.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _get_project_root() -> Path:
    """Resolve project root directory across training suite or datasets repositories."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "config.yaml").exists() or (parent / "unified_data.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent.parent


@dataclass(frozen=True)
class ModelNotebookMeta:
    """Immutable model metadata record consumed by notebook cell generators."""

    model_key: str
    pascal_name: str
    kebab_name: str
    is_forex: bool
    dataset_slug: str
    ds_list: list[str]
    ds_keys_repr: str
    clean_kaggle_repo: str
    primary_manifold: str
    k_username: str
    k_slug: str
    k_handle: str
    input_size: tuple[int, int]
    filename: str
    no_download: bool


def resolve_model_metadata(
    model_key: str,
    registry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ModelNotebookMeta:
    """Resolve comprehensive notebook generation metadata for a given model key."""
    pascal_name = model_key.replace("_", " ").title().replace(" ", "")
    kebab_name = model_key.replace("_", "-")

    dataset_slug = f"lemgendary-{kebab_name}"
    if config:
        k_urls = config.get("kaggle_dataset_urls", {})
        if isinstance(k_urls, dict):
            for key, url in k_urls.items():
                if pascal_name in key:
                    dataset_slug = url.split("/")[-1]
                    break
        elif isinstance(k_urls, list) and k_urls:
            dataset_slug = k_urls[0].split("/")[-1]

    model_info: dict[str, Any] = registry.get(model_key, {}) if registry else {}
    ds_raw = model_info.get("datasets", []) or model_info.get("dataset", [])
    if isinstance(ds_raw, str):
        ds_list = [ds_raw]
    elif isinstance(ds_raw, (list, tuple)):
        ds_list = list(ds_raw)
    else:
        ds_list = []

    is_forex = model_info.get("dataset_type") == "forex" or "forex" in model_key.lower()
    ds_keys_repr = repr(
        [model_key.lower(), model_key.replace("_", "-"), model_key.replace("_", "")]
        + [d.lower() for d in ds_list]
        + (["forex"] if is_forex else [])
    )

    k_username = config.get("kaggle_username", "lemtreursi") if config else "lemtreursi"
    slug_prefix = config.get("kaggle_slug_prefix", "lemgendary-") if config else "lemgendary-"
    slug_suffix = config.get("kaggle_slug_suffix", "-checkpoints") if config else "-checkpoints"

    k_slug = model_key.replace("_", "-")
    if "nima-aesthetic" in k_slug:
        k_slug = k_slug.replace("nima-aesthetic", "nima-aesthetics")

    k_handle = f"{k_username}/{slug_prefix}{k_slug}{slug_suffix}/pytorch/default"

    kaggle_ref = model_info.get("kaggle_ref", "")
    if not kaggle_ref:
        urls = model_info.get("kaggle_dataset_urls", [])
        if urls:
            kaggle_ref = urls[0]
    if kaggle_ref.startswith("kaggle://"):
        clean_kaggle_repo = kaggle_ref[len("kaggle://"):].strip()
    elif "kaggle.com/datasets/" in kaggle_ref:
        clean_kaggle_repo = kaggle_ref.split("kaggle.com/datasets/")[-1].strip().strip("/")
    else:
        clean_kaggle_repo = kaggle_ref.strip()

    primary_manifold = (
        ds_list[0]
        if ds_list
        else (f"LemGendized{pascal_name}Large" if not is_forex else "LemGendizedForexUniverseLarge")
    )
    if not clean_kaggle_repo:
        clean_kaggle_repo = f"lemtreursi/{primary_manifold.lower()}"

    no_download = bool(config.get("notebook_no_download", False)) if config else False

    filename = model_info.get("filename", model_key)
    size_raw = model_info.get("input_size", [3, 256, 256])
    if isinstance(size_raw, list):
        if len(size_raw) == 3:
            h, w = int(size_raw[1]), int(size_raw[2])
        elif len(size_raw) >= 2:
            h, w = int(size_raw[0]), int(size_raw[1])
        else:
            h, w = 256, 256
    elif isinstance(size_raw, int):
        h, w = size_raw, size_raw
    else:
        h, w = 256, 256

    return ModelNotebookMeta(
        model_key=model_key,
        pascal_name=pascal_name,
        kebab_name=kebab_name,
        is_forex=is_forex,
        dataset_slug=dataset_slug,
        ds_list=ds_list,
        ds_keys_repr=ds_keys_repr,
        clean_kaggle_repo=clean_kaggle_repo,
        primary_manifold=primary_manifold,
        k_username=k_username,
        k_slug=k_slug,
        k_handle=k_handle,
        input_size=(h, w),
        filename=filename,
        no_download=no_download,
    )


def load_registry(
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load neural network registry manifest (unified_models_v2.yaml)."""
    root = project_root if project_root is not None else _get_project_root()
    import yaml

    reg_name = "unified_models_v2.yaml"
    if config and "unified_models" in config:
        reg_name = str(config["unified_models"])

    reg_path = root / reg_name
    if not reg_path.exists():
        return {}

    with open(reg_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content if isinstance(content, dict) else {}
