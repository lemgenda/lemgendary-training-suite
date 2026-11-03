"""Kaggle dataset handles of a registry model, derived from its manifolds and the ecosystem owner.

``unified_models_v2.yaml`` lists each model's compiled manifolds under ``datasets``; the Kaggle
owner lives only in ``lemgendary-env-manager/ecosystem/ecosystem.json``. The dataset handle of a
manifold is therefore ``<owner>/<manifold lower-cased>``. A registry entry may still carry
``kaggle_ref`` / ``kaggle_dataset_urls`` as an explicit per-model override (for a manifold that
is published under another account or slug); when present they win over the derived handles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.ecosystem import kaggle_dataset_handle, kaggle_dataset_url

_URI_PREFIX = "kaggle://"
_WEB_MARKER = "kaggle.com/datasets/"


def clean_kaggle_handle(ref: str) -> str:
    """Normalise a ``kaggle://`` URI, Kaggle web URL or bare handle to ``<owner>/<slug>``."""
    text = ref.strip()
    if text.startswith(_URI_PREFIX):
        return text[len(_URI_PREFIX) :].strip().strip("/")
    if _WEB_MARKER in text:
        return text.split(_WEB_MARKER)[-1].strip().strip("/")
    return text


def model_manifolds(model_info: Mapping[str, Any]) -> list[str]:
    """Unique manifold names of a registry entry (``datasets``, or legacy ``dataset``), in order."""
    raw = model_info.get("datasets") or model_info.get("dataset") or []
    names = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else []
    return list(dict.fromkeys(str(n) for n in names if n))


def model_kaggle_dataset_handles(model_info: Mapping[str, Any]) -> list[str]:
    """Kaggle dataset handles of a registry entry: explicit override first, else derived."""
    explicit: list[str] = []
    ref = model_info.get("kaggle_ref")
    if isinstance(ref, str) and ref.strip():
        explicit.append(clean_kaggle_handle(ref))
    urls = model_info.get("kaggle_dataset_urls")
    if isinstance(urls, (list, tuple)):
        explicit.extend(clean_kaggle_handle(str(u)) for u in urls if str(u).strip())
    if explicit:
        return list(dict.fromkeys(explicit))
    return [kaggle_dataset_handle(name) for name in model_manifolds(model_info)]


def model_kaggle_dataset_urls(model_info: Mapping[str, Any]) -> list[str]:
    """Kaggle web URLs of :func:`model_kaggle_dataset_handles`."""
    return [kaggle_dataset_url(h) for h in model_kaggle_dataset_handles(model_info)]


def with_kaggle_dataset_refs(model_info: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of a registry entry with ``kaggle_ref`` / ``kaggle_dataset_urls`` filled in."""
    info = dict(model_info)
    handles = model_kaggle_dataset_handles(model_info)
    if handles:
        info["kaggle_ref"] = f"{_URI_PREFIX}{handles[0]}"
        info["kaggle_dataset_urls"] = [kaggle_dataset_url(h) for h in handles]
    return info
