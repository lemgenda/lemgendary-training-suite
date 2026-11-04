"""Read the shared LemGendary ecosystem file: service ports, projects and identity.

``lemgendary-env-manager/ecosystem/ecosystem.json`` is the single source of
truth for the facts every project shares (sidecar ports, project folders, the
Kaggle owner, the documentation URL). This module is the only place in this
repository that locates and parses it. Lookup order:

1. ``LEMGENDARY_ECOSYSTEM_FILE`` (tests point it at a fixture);
2. ``<workspace_root>/lemgendary-env-manager/ecosystem/ecosystem.json``, where
   the workspace root is ``LEMGENDARY_WORKSPACE_ROOT`` or this repository's
   parent folder (see :func:`training.utils.paths.get_workspace_root`).

Cloud notebooks clone ``lemgendary-env-manager`` next to the suite, so the
default location also resolves on Kaggle and Colab.

A missing, unreadable or malformed file raises :class:`EcosystemFileError`
naming the path that was expected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from training.utils.paths import WORKSPACE_ROOT_ENV, get_workspace_root

ECOSYSTEM_FILE_ENV = "LEMGENDARY_ECOSYSTEM_FILE"
"""Environment variable that overrides the ecosystem file location."""

ENV_MANAGER_FOLDER = "lemgendary-env-manager"
"""Folder that owns the ecosystem file (needed to find the file itself)."""

ENV_MANAGER_SERVICE = "env-manager"
DATASETS_SERVICE = "dataset-compiler"
TRAINING_SERVICE = "training-suite"


class EcosystemFileError(RuntimeError):
    """The ecosystem file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Service:
    """An HTTP sidecar entry of the ecosystem file."""

    id: str
    aliases: tuple[str, ...]
    project: str
    port: int
    health_path: str


@dataclass(frozen=True)
class Ecosystem:
    """The parts of the ecosystem file this repository uses."""

    path: Path
    docs_url: str
    kaggle_owner: str
    bind_host: str
    services: tuple[Service, ...]

    def service(self, service_id: str) -> Service:
        """Return the service whose id or alias is ``service_id`` (case-insensitive)."""
        wanted = service_id.strip().lower()
        for svc in self.services:
            if wanted == svc.id or wanted in svc.aliases:
                return svc
        raise EcosystemFileError(f"Unknown service '{service_id}' in {self.path}")


def ecosystem_file() -> Path:
    """Path of the ecosystem file that :func:`ecosystem` reads."""
    override = os.environ.get(ECOSYSTEM_FILE_ENV)
    if override:
        return Path(override)
    return get_workspace_root() / ENV_MANAGER_FOLDER / "ecosystem" / "ecosystem.json"


def _require(mapping: dict[str, Any], key: str, kind: type, path: Path) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise EcosystemFileError(f"Invalid ecosystem file {path}: '{key}' must be {kind.__name__}")
    return value


def _parse_service(raw: Any, path: Path) -> Service:
    if not isinstance(raw, dict):
        raise EcosystemFileError(f"Invalid ecosystem file {path}: service entry is not an object")
    aliases = _require(raw, "aliases", list, path)
    return Service(
        id=_require(raw, "id", str, path),
        aliases=tuple(str(a).lower() for a in aliases),
        project=_require(raw, "project", str, path),
        port=_require(raw, "port", int, path),
        health_path=_require(raw, "health_path", str, path),
    )


@cache
def _load(path: Path) -> Ecosystem:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EcosystemFileError(
            f"Ecosystem file not found: expected {path} (check out "
            f"{ENV_MANAGER_FOLDER} next to this repository, or set "
            f"{ECOSYSTEM_FILE_ENV} / {WORKSPACE_ROOT_ENV})"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EcosystemFileError(f"Cannot read ecosystem file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EcosystemFileError(f"Invalid ecosystem file {path}: top level is not an object")
    identity = _require(raw, "identity", dict, path)
    services = _require(raw, "services", list, path)
    return Ecosystem(
        path=path,
        docs_url=_require(identity, "docs_url", str, path).rstrip("/"),
        kaggle_owner=_require(identity, "kaggle_owner", str, path),
        bind_host=_require(identity, "bind_host", str, path),
        services=tuple(_parse_service(svc, path) for svc in services),
    )


def ecosystem() -> Ecosystem:
    """Return the parsed ecosystem file, cached per path."""
    return _load(ecosystem_file())


def clear_cache() -> None:
    """Forget every parsed file (tests that switch fixtures call this)."""
    _load.cache_clear()


def service_port(service_id: str) -> int:
    """TCP port of the sidecar ``service_id`` (id or alias, e.g. ``"training"``)."""
    return ecosystem().service(service_id).port


def service_url(service_id: str) -> str:
    """Base HTTP URL (``http://<bind_host>:<port>``) of the sidecar ``service_id``."""
    eco = ecosystem()
    return f"http://{eco.bind_host}:{eco.service(service_id).port}"


def bind_host() -> str:
    """Loopback host every sidecar binds to."""
    return ecosystem().bind_host


def kaggle_owner() -> str:
    """Kaggle account that owns the published manifolds and model checkpoints."""
    return ecosystem().kaggle_owner


def docs_url() -> str:
    """Base URL of the documentation hub (no trailing slash)."""
    return ecosystem().docs_url


def project_dir(service_id: str) -> Path:
    """Checkout folder of the project that runs ``service_id``."""
    return get_workspace_root() / ecosystem().service(service_id).project


def kaggle_dataset_handle(manifold: str) -> str:
    """Kaggle dataset handle (``<owner>/<slug>``) of a manifold such as ``LemGendizedUpnV2``."""
    return f"{kaggle_owner()}/{manifold.strip().lower()}"


def kaggle_dataset_url(handle: str) -> str:
    """Public Kaggle web URL of the dataset ``handle`` (``<owner>/<slug>``)."""
    return f"https://www.kaggle.com/datasets/{handle}"
