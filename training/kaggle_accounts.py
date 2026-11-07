"""Kaggle account registry, authentication and kernel and model lookups."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from training.cloud.credentials import (
    load_kaggle_users as load_vault_users,
    parse_kaggle_users_text,
)
from training.cloud.errors import KAGGLE_API_ERRORS
from training.utils.lazy_imports import load_kaggle_api

logger = logging.getLogger("lemtrain.cloud.kaggle_monitor")


def find_kaggle_users_file(explicit_path: Path | None = None) -> Path | None:
    """
    Locates the .kaggle_users credentials registry file.
    Searches explicit path, local suite directory, working directory, and sibling directories.
    """
    if explicit_path and explicit_path.exists():
        return explicit_path

    candidates = [
        Path.cwd() / ".kaggle_users",
        Path(__file__).resolve().parent.parent / ".kaggle_users",
        Path.cwd().parent / "lemgendary-training-suite" / ".kaggle_users",
        Path(__file__).resolve().parent.parent.parent
        / "lemgendary-training-suite"
        / ".kaggle_users",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return Path(__file__).resolve().parent.parent / ".kaggle_users"


def load_kaggle_users(users_file: Path | None = None) -> tuple[list[dict[str, str]], Path]:
    """
    Parses configured Kaggle user accounts, prioritizing .secrets.yaml.
    """
    vault_accounts = load_vault_users(users_file)
    if vault_accounts:
        target_file = find_kaggle_users_file(users_file) or (Path.cwd() / ".kaggle_users")
        clean_accounts = [
            {"username": str(a.get("username", "")), "token": str(a.get("token", ""))}
            for a in vault_accounts
        ]
        return clean_accounts, target_file

    target_file = find_kaggle_users_file(users_file)
    accounts: list[dict[str, str]] = []

    if target_file and target_file.exists():
        try:
            content = target_file.read_text(encoding="utf-8")
            accounts.extend(
                {"username": user, "token": token}
                for user, token in parse_kaggle_users_text(content)
            )
        except OSError as e:
            print(f"[WARN] Failed to read users registry at {target_file}: {e}")

    return accounts, target_file or (Path.cwd() / ".kaggle_users")


def save_kaggle_user(users_file: Path, username: str, token: str) -> bool:
    """
    Appends a new user account definition to the .kaggle_users file.
    """
    try:
        users_file.parent.mkdir(parents=True, exist_ok=True)
        with open(users_file, "a", encoding="utf-8") as f:
            f.write(f"\nKAGGLE_USERNAME={username}, KAGGLE_API_TOKEN={token};")
        return True
    except OSError as e:
        print(f"[ERROR] Failed to save user to {users_file}: {e}")
        return False


def authenticate_kaggle_user(username: str, token: str) -> Any | None:
    """
    Configures environment variables and initializes authenticated KaggleApi instance.
    """
    clean_token = token.replace("KGAT_", "").strip().rstrip(";").strip()

    # Clear conflicting access token variables that trigger OAuth introspection
    os.environ.pop("KAGGLE_API_TOKEN", None)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = clean_token

    try:
        # Deferred: the Kaggle SDK is optional and reads credentials at import time.
        api = load_kaggle_api()()
        api.authenticate()
        return api
    except (ImportError, *KAGGLE_API_ERRORS) as exc:
        print(f"\n[ERROR] Authentication failed for user '{username}': {exc}")
        print("[REMEDY] Verify username and API token validity on kaggle.com/settings/api.")
        return None


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
