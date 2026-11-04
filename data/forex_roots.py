"""Forex shard root discovery for local, Colab and Kaggle runs (with a Google Drive fallback)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import zipfile

import yaml

logger = logging.getLogger("lemtrain.data.forex_dataset")

_COLAB_BASE = "/content/LemGendaryDatasets"
_COLAB_DRIVE = "/content/drive/MyDrive/LemGendaryDatasets"
_KAGGLE_INPUT = "/kaggle/input"


def _suite_root() -> str:
    """Return the training suite directory (the parent of ``data/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _datasets_hub() -> str:
    """Return the sibling ``LemGendaryDatasets`` directory."""
    return os.path.abspath(os.path.join(_suite_root(), "..", "LemGendaryDatasets"))


def _gdrive_dataset_ids() -> list:
    """Return ``forex_predictor.google_drive_dataset_ids`` from the model registry YAML."""
    unified_path = os.path.join(_suite_root(), "unified_models_v2.yaml")
    if not os.path.exists(unified_path):
        return []
    try:
        with open(unified_path, "r", encoding="utf-8") as f:
            um = yaml.safe_load(f)
            return um.get("forex_predictor", {}).get("google_drive_dataset_ids", [])
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        # AttributeError: the YAML root (or forex_predictor entry) is not a mapping.
        logger.warning("Could not read google_drive_dataset_ids from %s: %s", unified_path, exc)
    return []


def _detect_env() -> str:
    """Return ``colab``, ``kaggle`` or the ``ENV`` variable (default ``local``)."""
    env = os.environ.get("ENV", "local")
    if "colab" in sys.modules or os.path.exists("/content"):
        env = "colab"
    elif os.path.exists("/kaggle"):
        env = "kaggle"
    return env


def _add_forex_entries(base: str, entries: list[str], roots: list[str]) -> None:
    """Add each ``*forex*`` entry of ``base`` (its ``forex/`` child when present) to roots."""
    for d in entries:
        if "forex" in d.lower():
            cand_forex = os.path.join(base, d, "forex")
            cand = cand_forex if os.path.isdir(cand_forex) else os.path.join(base, d)
            if os.path.isdir(cand) and cand not in roots:
                roots.append(cand)


def _add_colab_roots(roots: list[str]) -> None:
    """Add the Colab dataset packages; exit when none are attached."""
    for base in (_COLAB_BASE, _COLAB_DRIVE):
        if os.path.exists(base):
            _add_forex_entries(base, os.listdir(base), roots)
    if not roots:
        print(f"\n[ERROR] Colab requires dataset packages at {_COLAB_BASE} or {_COLAB_DRIVE}")
        sys.exit(1)


def _add_kaggle_roots(roots: list[str]) -> None:
    """Add every ``forex`` directory under the attached Kaggle inputs."""
    if os.path.exists(_KAGGLE_INPUT):
        for root_dir, dirs, _ in os.walk(_KAGGLE_INPUT):
            if "forex" in dirs:
                p = os.path.join(root_dir, "forex")
                if p not in roots:
                    roots.append(p)
            for d in dirs:
                if "forex" in d.lower():
                    p = os.path.join(root_dir, d)
                    if p not in roots:
                        roots.append(p)
    if not roots:
        print("\n[WARNING] No attached forex datasets found in Kaggle /kaggle/input!")


def _download_gdrive_roots(gdrive_ids: list, roots: list[str]) -> None:
    """Download and unpack the Google Drive dataset packages, then add their forex roots."""
    print(
        "\n[DATA] Local forex dataset not found. Attempting gdown from "
        f"{len(gdrive_ids)} Google Drive IDs..."
    )
    base_out = _datasets_hub()
    os.makedirs(base_out, exist_ok=True)
    for gid in gdrive_ids:
        dest = os.path.join(base_out, f"dataset_{gid}.zip")
        subprocess.run([sys.executable, "-m", "gdown", "--id", gid, "-O", dest], check=False)
        if os.path.exists(dest):
            with zipfile.ZipFile(dest, "r") as zip_ref:
                zip_ref.extractall(base_out)
            os.remove(dest)
    for d in os.listdir(base_out):
        if "forex" in d.lower():
            cand = (
                os.path.join(base_out, d, "forex")
                if os.path.isdir(os.path.join(base_out, d, "forex"))
                else os.path.join(base_out, d)
            )
            if os.path.exists(cand) and cand not in roots:
                roots.append(cand)


def _add_local_roots(roots: list[str], gdrive_ids: list) -> None:
    """Add the local dataset hub and ``data/`` forex packages, else download them."""
    for bs in (_datasets_hub(), os.path.abspath(os.path.join(_suite_root(), "data"))):
        if not os.path.exists(bs):
            continue
        try:
            entries = os.listdir(bs)
        except OSError as exc:
            logger.warning("Skipping unreadable dataset search root %s: %s", bs, exc)
            continue
        _add_forex_entries(bs, entries, roots)

    if not roots and gdrive_ids:
        _download_gdrive_roots(gdrive_ids, roots)


def discover_shard_roots(shard_root: str) -> list[str]:
    """Return the Forex shard roots: ``shard_root`` plus the packages found for this runtime."""
    gdrive_ids = _gdrive_dataset_ids()
    env = _detect_env()

    roots: list[str] = []
    if shard_root and os.path.exists(shard_root):
        roots.append(os.path.abspath(shard_root))

    if env == "colab":
        _add_colab_roots(roots)
    elif env == "kaggle":
        _add_kaggle_roots(roots)
    else:  # Local
        _add_local_roots(roots, gdrive_ids)
    return roots
