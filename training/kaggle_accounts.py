"""Kaggle vault accounts and kernel and model lookups.

Authentication lives in :mod:`training.cloud.kaggle_client`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from training.cloud.credentials import load_kaggle_users
from training.cloud.errors import KAGGLE_API_ERRORS

logger = logging.getLogger("lemtrain.cloud.kaggle_monitor")


def vault_accounts() -> list[dict[str, str]]:
    """Kaggle accounts (``username``, ``token``) of the ``.secrets.yaml`` vault, default first.

    The vault (managed by the environment manager) is the only account registry; the legacy
    ``.kaggle_users`` file is neither read nor written (K-AUTH-4).
    """
    return [
        {"username": str(a.get("username", "")), "token": str(a.get("token", ""))}
        for a in load_kaggle_users()
    ]


def fetch_user_kernels(api: Any, username: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Queries active and recent kernels for the target Kaggle user with concurrent status resolution.
    """
    try:
        raw_kernels = api.kernels_list(user=username, page_size=limit, sort_by="dateRun")
    except KAGGLE_API_ERRORS as exc:
        print(f"[ERROR] Failed to list kernels for {username}: {exc}")
        return []

    if not raw_kernels:
        return []

    def _resolve_status(kernel_item: Any) -> dict[str, Any]:
        ref = getattr(kernel_item, "ref", str(kernel_item))
        title = getattr(kernel_item, "title", ref)
        status_val = "UNKNOWN"
        failure_msg = None
        try:
            st = api.kernels_status(ref)
            status_val = str(getattr(st, "status", st)).rsplit(".", maxsplit=1)[-1].upper()
            failure_msg = getattr(st, "failure_message", None) or getattr(
                st, "failureMessage", None
            )
        except KAGGLE_API_ERRORS as exc:
            logger.debug("Could not resolve status for kernel %s: %s", ref, exc)
        return {"ref": ref, "title": title, "status": status_val, "failure_message": failure_msg}

    with ThreadPoolExecutor(max_workers=5) as executor:
        kernels_data = list(executor.map(_resolve_status, raw_kernels))

    return kernels_data


def get_registered_models() -> list[str]:
    """Loads model manifold names registered in unified_models_v2.yaml."""
    yaml_path = Path(__file__).resolve().parent.parent / "unified_models_v2.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return [k for k in data if k != "_registry_metadata"]
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.warning("Could not load registered models from %s: %s", yaml_path, exc)
    return []


def match_model_from_slug(slug: str | None) -> str | None:
    """Infers registered model manifold key from kernel slug."""
    if not slug:
        return None
    models = get_registered_models()
    clean_slug = slug.lower().replace("-", "").replace("_", "")
    for m in models:
        clean_m = m.replace("_", "").lower()
        if clean_m in clean_slug:
            return m
    if "forex" in clean_slug:
        return "forex_predictor"
    return None
