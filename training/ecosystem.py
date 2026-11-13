"""Read the shared LemGendary ecosystem file: service ports, projects and identity.

``lemgendary-env-manager/ecosystem/ecosystem.json`` is the single source of
truth for the facts every project shares (sidecar ports, project folders, the
Kaggle owner, the documentation URL, the GitHub owner URL and the ``kaggle``
section: model collaborators and accelerators). The readers here mirror those of
``env_manager.ecosystem`` (this suite does not install the environment manager).
This module is the only place in this repository that locates and parses it.
Lookup order:

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


_ACCELERATOR_KINDS = ("cpu", "gpu", "tpu")


@dataclass(frozen=True)
class KaggleAccelerator:
    """A Kaggle session type (``machine_shape`` is None exactly for CPU)."""

    id: str
    label: str
    kind: str
    machine_shape: str | None


@dataclass(frozen=True)
class KaggleSettings:
    """The ``kaggle`` section of the ecosystem file."""

    collaborators: tuple[str, ...]
    """Kaggle usernames granted the editor role on pushed models."""
    default_accelerator: str
    accelerators: tuple[KaggleAccelerator, ...]
    retired_machine_shapes: tuple[str, ...]

    def accelerator(self, accelerator_id: str) -> KaggleAccelerator:
        """The accelerator ``accelerator_id``; raise :class:`EcosystemFileError` when unknown."""
        for acc in self.accelerators:
            if acc.id == accelerator_id:
                return acc
        raise EcosystemFileError(f"Unknown Kaggle accelerator '{accelerator_id}'")


@dataclass(frozen=True)
class GitHubSettings:
    """Where the workspace projects are hosted."""

    url: str
    """``identity.github_url`` without a trailing slash."""
    project_folders: tuple[str, ...]
    """Folders of the ``projects`` list (each is a repository name)."""


@dataclass(frozen=True)
class Ecosystem:
    """The parts of the ecosystem file this repository uses."""

    path: Path
    docs_url: str
    kaggle_owner: str
    bind_host: str
    github: GitHubSettings
    services: tuple[Service, ...]
    kaggle: KaggleSettings

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


def _object(raw: Any, path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EcosystemFileError(f"Invalid ecosystem file {path}: list entry is not an object")
    return raw


def _strings(raw: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    values = _require(raw, key, list, path)
    if not all(isinstance(v, str) for v in values):
        raise EcosystemFileError(f"Invalid ecosystem file {path}: '{key}' must list strings")
    return tuple(values)


def _parse_accelerator(raw: Any, path: Path) -> KaggleAccelerator:
    entry = _object(raw, path)
    shape = entry.get("machine_shape")
    acc = KaggleAccelerator(
        id=_require(entry, "id", str, path),
        label=_require(entry, "label", str, path),
        kind=_require(entry, "kind", str, path),
        machine_shape=shape if isinstance(shape, str) else None,
    )
    if acc.kind not in _ACCELERATOR_KINDS:
        raise EcosystemFileError(f"Invalid ecosystem file {path}: accelerator {acc.id} kind")
    if (acc.kind == "cpu") != (acc.machine_shape is None):
        raise EcosystemFileError(
            f"Invalid ecosystem file {path}: accelerator {acc.id} machine_shape is null "
            "exactly for cpu"
        )
    return acc


def _parse_kaggle(raw: dict[str, Any], path: Path) -> KaggleSettings:
    settings = KaggleSettings(
        collaborators=_strings(raw, "collaborators", path),
        default_accelerator=_require(raw, "default_accelerator", str, path),
        accelerators=tuple(
            _parse_accelerator(acc, path) for acc in _require(raw, "accelerators", list, path)
        ),
        retired_machine_shapes=_strings(raw, "retired_machine_shapes", path),
    )
    ids = [acc.id for acc in settings.accelerators]
    if len(set(ids)) != len(ids) or settings.default_accelerator not in ids:
        raise EcosystemFileError(
            f"Invalid ecosystem file {path}: accelerator ids must be unique and include "
            f"default_accelerator '{settings.default_accelerator}'"
        )
    retired = {shape.lower() for shape in settings.retired_machine_shapes}
    for acc in settings.accelerators:
        if acc.machine_shape and acc.machine_shape.lower() in retired:
            raise EcosystemFileError(
                f"Invalid ecosystem file {path}: accelerator {acc.id} uses retired "
                f"{acc.machine_shape}"
            )
    return settings


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
        github=GitHubSettings(
            url=_require(identity, "github_url", str, path).rstrip("/"),
            project_folders=tuple(
                _require(_object(proj, path), "folder", str, path)
                for proj in _require(raw, "projects", list, path)
            ),
        ),
        services=tuple(_parse_service(svc, path) for svc in services),
        kaggle=_parse_kaggle(_require(raw, "kaggle", dict, path), path),
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


def kaggle_collaborators() -> tuple[str, ...]:
    """Kaggle usernames that get the editor role on pushed models (``kaggle.collaborators``)."""
    return ecosystem().kaggle.collaborators


def kaggle_accelerators() -> tuple[KaggleAccelerator, ...]:
    """Every Kaggle session type, in the order a picker should show them."""
    return ecosystem().kaggle.accelerators


def kaggle_accelerator(accelerator_id: str | None = None) -> KaggleAccelerator:
    """The accelerator ``accelerator_id``, or the default one when ``None``."""
    settings = ecosystem().kaggle
    return settings.accelerator(accelerator_id or settings.default_accelerator)


def kaggle_retired_machine_shapes() -> tuple[str, ...]:
    """Machine shapes Kaggle no longer offers (compare case-insensitively)."""
    return ecosystem().kaggle.retired_machine_shapes


def github_url() -> str:
    """GitHub owner URL (``identity.github_url``, no trailing slash)."""
    return ecosystem().github.url


def github_repo_url(folder: str) -> str:
    """Plain (credential-free) git clone URL of the workspace project in ``folder``."""
    eco = ecosystem()
    if folder not in eco.github.project_folders:
        raise EcosystemFileError(f"Unknown project '{folder}' in {eco.path}")
    return f"{eco.github.url}/{folder}.git"
