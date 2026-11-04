"""Resolve and validate the inputs of a FID re-evaluation (model, checkpoint, validation data).

Everything here runs on CPU and is shared by ``--dry-run`` and the real run, so a dry run proves
the checkpoint loads into the registered architecture and the validation split has images.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from data.dataset import MultiTaskDataset
from models.factory import get_model
from training.checkpoint.manager import CHECKPOINT_LOAD_ERRORS, safe_load_checkpoint
from training.context_data import DATASET_BUILD_ERRORS, LLM_TASKS
from training.fid_reeval.measure import MIN_IMAGES

_STATE_KEYS = ("model_state", "model_state_dict", "state_dict", "model")
_STRIP_PREFIXES = ("module.", "_orig_mod.")
_NON_IMAGE_TASKS = (*LLM_TASKS, "forex", "quality")


class InputError(ValueError):
    """An input of the re-evaluation is missing or unusable."""


@dataclass(frozen=True)
class ReevalRequest:
    """What the user asked to evaluate."""

    model_id: str
    checkpoint: Path
    datasets_root: Path | None
    env: str = "local"
    strict: bool = True


@dataclass
class ResolvedInputs:
    """Validated inputs, ready for the GPU pass."""

    request: ReevalRequest
    config: dict[str, Any]
    model: torch.nn.Module
    dataset: Any
    checkpoint_epoch: str
    checkpoint_sha256: str
    load_report: str


def load_config(project_root: Path) -> dict[str, Any]:
    """Workspace ``config.yaml``."""
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def registry_entry(project_root: Path, config: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Registry entry of ``model_id``; it must be an image model with a ``fid`` SOTA target."""
    path = project_root / config.get("unified_models", "unified_models_v2.yaml")
    with open(path, encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}
    nested = registry.get("models")
    models = nested if isinstance(nested, dict) else registry
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        raise InputError(f"unknown model id {model_id!r} (not in {path.name})")
    if "fid" not in (entry.get("sota_targets") or {}):
        raise InputError(f"{model_id!r} has no 'fid' SOTA target; FID is not one of its metrics")
    task = entry.get("dataset_type", "restoration")
    task = task[0] if isinstance(task, list) else task
    if task in _NON_IMAGE_TASKS:
        raise InputError(f"{model_id!r} is a {task!r} model; FID needs image outputs")
    return entry


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Model weights of a checkpoint payload with DDP / ``torch.compile`` prefixes removed."""
    state = next((payload[k] for k in _STATE_KEYS if isinstance(payload.get(k), dict)), payload)
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise InputError("checkpoint holds no tensor state dict under " + ", ".join(_STATE_KEYS))
    cleaned = {}
    for key, value in state.items():
        for prefix in _STRIP_PREFIXES:
            key = key.removeprefix(prefix)
        cleaned[key] = value
    return cleaned


def load_weights(model: torch.nn.Module, state: dict[str, Any], strict: bool) -> str:
    """Load ``state`` into ``model``; strict mode refuses any missing or unexpected key."""
    result = model.load_state_dict(state, strict=False)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    report = f"missing={len(missing)} unexpected={len(unexpected)}"
    if strict and (missing or unexpected):
        raise InputError(
            f"checkpoint does not match the architecture ({report}; first missing "
            f"{missing[:5]}, first unexpected {unexpected[:5]}); pass --non-strict only if "
            "you know why"
        )
    return ("strict" if strict else "non-strict") + " load, " + report


def build_model(model_id: str, config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the registered architecture of ``model_id`` (CPU, random weights)."""
    return get_model(model_id, config)


def build_val_dataset(
    model_id: str, config: dict[str, Any], datasets_root: Path | None, env: str
) -> Any:
    """Validation split of ``model_id``'s manifold, built exactly like training builds it."""
    cfg = copy.deepcopy(config)
    if datasets_root is not None:
        cfg.setdefault("paths", {})["datasets_root"] = str(datasets_root)
    return MultiTaskDataset(cfg, model_key=model_id, is_train=False, env=env, sample_fraction=1.0)


def _checked_paths(request: ReevalRequest) -> None:
    if not request.checkpoint.is_file():
        raise InputError(f"checkpoint not found: {request.checkpoint}")
    if request.datasets_root is not None and not request.datasets_root.is_dir():
        raise InputError(f"datasets root is not a directory: {request.datasets_root}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return safe_load_checkpoint(path, map_location="cpu")
    except CHECKPOINT_LOAD_ERRORS as exc:
        raise InputError(f"checkpoint {path} cannot be loaded: {exc}") from exc


def _dataset(request: ReevalRequest, config: dict[str, Any]) -> Any:
    try:
        dataset = build_val_dataset(request.model_id, config, request.datasets_root, request.env)
    except DATASET_BUILD_ERRORS as exc:
        raise InputError(f"validation split cannot be built: {exc}") from exc
    if len(dataset) < MIN_IMAGES:
        raise InputError(
            f"validation split of {request.model_id!r} has {len(dataset)} images "
            f"(need at least {MIN_IMAGES}); check --datasets-root"
        )
    return dataset


def resolve_inputs(request: ReevalRequest, project_root: Path) -> ResolvedInputs:
    """Validate every input; raises :class:`InputError` on the first problem."""
    config = load_config(project_root)
    registry_entry(project_root, config, request.model_id)
    _checked_paths(request)
    payload = _load_checkpoint(request.checkpoint)
    model = build_model(request.model_id, config)
    report = load_weights(model, checkpoint_state(payload), request.strict)
    return ResolvedInputs(
        request=request,
        config=config,
        model=model,
        dataset=_dataset(request, config),
        checkpoint_epoch=str(payload.get("epoch", "")),
        checkpoint_sha256=sha256_file(request.checkpoint),
        load_report=report,
    )
