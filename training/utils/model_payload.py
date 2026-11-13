"""The one loader of model weight files: release ``.pt`` payloads, checkpoints, state dicts.

A model file is one of:

* a release payload written by :func:`training.export.torch_standalone.export_torch_standalone`:
  ``{"model_state": state_dict, "model": nn.Module, "metadata": {...}}``;
* a training checkpoint (``.pth``): the weights under ``model_state`` (older:
  ``model_state_dict`` / ``state_dict``) next to optimizer, scheduler and metrics state;
* a bare state dict (pre-R2-T1 releases, the LLM release ``.pt``);
* a pickled ``nn.Module``.

:func:`model_payload_of` normalizes an already-loaded object and :func:`load_model_payload`
reads a file. Loading tries ``torch.load(weights_only=True)`` first, which is the torch >=
2.6 default and reads bare state dicts safely. A file that pickles Python objects (the
release ``model`` entry, optimizer state of a checkpoint) is only unpickled in full when
the caller passes ``trust_pickle=True`` (the suite's own artifacts); otherwise
:class:`UntrustedPickleError` is raised. The pickled ``model`` is optional: consumers
should load ``state_dict`` into a model they build.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

import torch
from torch import nn

logger = logging.getLogger("lemtrain.model_payload")

STATE_KEYS: tuple[str, ...] = ("model_state", "model_state_dict", "state_dict")
"""Payload keys holding the weights, in precedence order."""


class ModelPayloadError(ValueError):
    """The object or file holds no model weights."""


class UntrustedPickleError(ModelPayloadError):
    """The file needs full unpickling, which the caller did not trust."""


@dataclass(frozen=True)
class ModelPayload:
    """Weights of a model file plus whatever else the file carried."""

    state_dict: dict[str, torch.Tensor]
    kind: str
    """``release``, ``checkpoint``, ``state_dict`` or ``module``."""
    module: nn.Module | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    """The whole payload dict (empty for a bare state dict or a pickled module)."""


def _is_state_dict(obj: Any) -> TypeGuard[dict[str, torch.Tensor]]:
    return (
        isinstance(obj, dict)
        and bool(obj)
        and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items())
    )


def model_payload_of(obj: Any) -> ModelPayload:
    """Normalize a loaded release payload, checkpoint, state dict or module."""
    if isinstance(obj, nn.Module):
        return ModelPayload(state_dict=dict(obj.state_dict()), kind="module", module=obj)
    if not isinstance(obj, dict):
        raise ModelPayloadError(f"expected a dict or nn.Module, got {type(obj).__name__}")
    for key in STATE_KEYS:
        state = obj.get(key)
        if isinstance(state, dict):
            module = obj.get("model")
            metadata = obj.get("metadata")
            is_release = isinstance(module, nn.Module) or isinstance(metadata, dict)
            return ModelPayload(
                state_dict=dict(state),
                kind="release" if is_release and key == "model_state" else "checkpoint",
                module=module if isinstance(module, nn.Module) else None,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
                raw=obj,
            )
    module = obj.get("model")
    if isinstance(module, nn.Module):
        return model_payload_of(module)
    if _is_state_dict(module):
        return ModelPayload(state_dict=dict(module), kind="checkpoint", raw=obj)
    if _is_state_dict(obj):
        return ModelPayload(state_dict=dict(obj), kind="state_dict")
    raise ModelPayloadError("no model weights under " + ", ".join(STATE_KEYS) + " or 'model'")


def load_model_payload(
    path: Path | str, map_location: Any = "cpu", *, trust_pickle: bool = False
) -> ModelPayload:
    """Read a model file; see the module docstring for the ``weights_only`` policy."""
    file_path = Path(path)
    try:
        loaded = torch.load(str(file_path), map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as safe_err:
        if not trust_pickle:
            raise UntrustedPickleError(
                f"{file_path.name} pickles Python objects and cannot be read with "
                f"weights_only=True; load it with trust_pickle=True only if it is a trusted "
                f"artifact ({safe_err})"
            ) from safe_err
        logger.info(
            "%s pickles Python objects (release module or optimizer state); "
            "loading the trusted file with weights_only=False.",
            file_path.name,
        )
        loaded = torch.load(str(file_path), map_location=map_location, weights_only=False)
    return model_payload_of(loaded)


EPOCH_READ_ERRORS: tuple[type[BaseException], ...] = (
    ModelPayloadError,
    OSError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
)
"""Failures of reading a model file that mean "no readable epoch" to :func:`checkpoint_epoch`."""


def checkpoint_epoch(path: Path | str) -> int | None:
    """The ``epoch`` stored in a suite checkpoint, or None when the file has none or is unreadable.

    Checkpoints are the suite's own artifacts (the local hub or the owner's Kaggle model), so
    they are loaded with ``trust_pickle=True``. Callers compare checkpoints by this value,
    never by file size.
    """
    try:
        payload = load_model_payload(path, trust_pickle=True)
    except EPOCH_READ_ERRORS as exc:
        logger.warning("Cannot read the epoch of %s: %s", Path(path).name, exc)
        return None
    epoch = payload.raw.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
        return None
    return int(epoch)


def pt_load_source(path_expr: str, device_expr: str) -> list[str]:
    """Source lines of generated code (notebooks, model cards, READMEs) loading a model file.

    The generated code applies the same policy as :func:`load_model_payload` without
    importing the suite: ``weights_only=True`` first, full unpickling only for the trusted
    release file, then ``model`` (the pickled module, or ``None``) and ``state`` (the state
    dict) for every payload shape. Lines end in newlines and carry no indentation.
    """
    return [
        "import pickle\n",
        "try:\n",
        f"    ckpt = torch.load({path_expr}, map_location={device_expr}, weights_only=True)\n",
        "except pickle.UnpicklingError:\n",
        "    # Release .pt files also pickle the nn.Module: unpickle trusted files only.\n",
        f"    ckpt = torch.load({path_expr}, map_location={device_expr}, weights_only=False)\n",
        "if isinstance(ckpt, torch.nn.Module):\n",
        "    pt_model, state = ckpt, ckpt.state_dict()\n",
        "elif 'model_state' in ckpt:\n",
        "    pt_model, state = ckpt.get('model'), ckpt['model_state']\n",
        "else:\n",
        "    pt_model, state = None, ckpt\n",
    ]
