"""Pareto Model Soup service shared by the CLI and the sidecar API.

Resolves a model's vault checkpoints (``vault_<metric>.pth``) inside the
configured LemGendaryModels hub and hands the list of files to
``training.export.soup.create_pareto_model_soup``.

The hub location has one source: ``paths.checkpoints_root`` in
``config.yaml``, resolved relative to the training-suite project root
(default ``../LemGendaryModels``, the sibling repo). This module does not
import torch; the blending function is imported only when a soup is built,
so path resolution and request validation can be tested in isolation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODELS_ROOT = "../LemGendaryModels"
VAULT_PREFIX = "vault_"
VAULT_SUFFIX = ".pth"
DEFAULT_SOUP_FILENAME = "best_soup.pth"
MIN_SOUP_DONORS = 2

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class SoupError(Exception):
    """Base error for Model Soup requests. ``http_status`` maps it to the API."""

    http_status = 400


class SoupCheckpointsMissing(SoupError):
    """The model's checkpoint folder or requested vault checkpoints do not exist."""

    http_status = 404

    def __init__(self, message: str, missing: Sequence[str]) -> None:
        super().__init__(message)
        self.missing = list(missing)


class SoupInvalidRequest(SoupError):
    """The request cannot be blended (fewer than two donors, bad weights, bad names)."""

    http_status = 422


@dataclass(frozen=True)
class SoupResult:
    """Outcome of a successful Pareto Model Soup blend."""

    model_key: str
    donors: list[Path]
    weights: list[float] | None
    output_path: Path


SoupFunction = Callable[..., dict[str, Any]]


def _validate_name(value: str, kind: str) -> str:
    if not _SAFE_NAME.match(value) or ".." in value:
        raise SoupInvalidRequest(f"Invalid {kind} '{value}': use letters, digits, '_', '-' or '.'.")
    return value


def resolve_models_root(project_root: Path) -> Path:
    """Return the LemGendaryModels hub folder configured in ``config.yaml``."""
    rel = DEFAULT_MODELS_ROOT
    cfg_path = project_root / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        configured = (cfg.get("paths") or {}).get("checkpoints_root")
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError(
                    "config.yaml 'paths.checkpoints_root' must be a non-empty relative path string."
                )
            if Path(configured).is_absolute():
                raise ValueError(
                    "config.yaml 'paths.checkpoints_root' must be relative to the project root."
                )
            rel = configured
    return (project_root / rel).resolve()


def resolve_checkpoints_dir(model_key: str, project_root: Path) -> Path:
    """Return ``<models root>/<model_key>/checkpoints`` for a validated model key."""
    _validate_name(model_key, "model key")
    return resolve_models_root(project_root) / model_key / "checkpoints"


def _vault_filename(donor: str) -> str:
    metric = donor.strip().lower().replace("-", "_")
    metric = metric.removeprefix(VAULT_PREFIX)
    metric = metric.removesuffix(VAULT_SUFFIX)
    _validate_name(metric, "donor")
    return f"{VAULT_PREFIX}{metric}{VAULT_SUFFIX}"


def resolve_vault_checkpoints(
    model_key: str,
    project_root: Path,
    donors: Sequence[str] | None = None,
) -> list[Path]:
    """Resolve the vault checkpoint files to blend for ``model_key``.

    Args:
        model_key: Model folder name inside the models hub.
        project_root: Training-suite project root (holds ``config.yaml``).
        donors: Optional vault metrics to blend (``psnr``, ``lpips`` or
            ``vault_psnr.pth``). When omitted, every ``vault_*.pth`` in the
            model's checkpoint folder is used, sorted by name.

    Raises:
        SoupCheckpointsMissing: the checkpoint folder or a requested vault is missing.
        SoupInvalidRequest: fewer than two donors, duplicates, or unsafe names.
    """
    ckpt_dir = resolve_checkpoints_dir(model_key, project_root)
    shown_dir = f"{ckpt_dir.parent.parent.name}/{model_key}/checkpoints"
    if not ckpt_dir.is_dir():
        raise SoupCheckpointsMissing(
            f"Checkpoint folder not found for model '{model_key}': {shown_dir}",
            missing=[shown_dir],
        )

    if donors:
        names = [_vault_filename(d) for d in donors]
        if len(set(names)) != len(names):
            raise SoupInvalidRequest(f"Duplicate donors in request: {', '.join(names)}")
        if len(names) < MIN_SOUP_DONORS:
            raise SoupInvalidRequest(
                f"Model Soup needs at least {MIN_SOUP_DONORS} vault checkpoints, got {len(names)}: {', '.join(names)}"
            )
        paths = [ckpt_dir / n for n in names]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise SoupCheckpointsMissing(
                f"Missing vault checkpoints for model '{model_key}' in {shown_dir}: {', '.join(missing)}",
                missing=missing,
            )
        return paths

    paths = sorted(p for p in ckpt_dir.glob(f"{VAULT_PREFIX}*{VAULT_SUFFIX}") if p.is_file())
    if not paths:
        expected = [f"{VAULT_PREFIX}psnr{VAULT_SUFFIX}", f"{VAULT_PREFIX}lpips{VAULT_SUFFIX}"]
        raise SoupCheckpointsMissing(
            f"No vault checkpoints found for model '{model_key}' in {shown_dir} "
            f"(expected e.g. {', '.join(expected)}).",
            missing=expected,
        )
    if len(paths) < MIN_SOUP_DONORS:
        raise SoupInvalidRequest(
            f"Model Soup needs at least {MIN_SOUP_DONORS} vault checkpoints; "
            f"found only {', '.join(p.name for p in paths)} in {shown_dir}."
        )
    return paths


def _validate_weights(weights: Sequence[float] | None, donor_count: int) -> list[float] | None:
    if weights is None:
        return None
    values = [float(w) for w in weights]
    if len(values) != donor_count:
        raise SoupInvalidRequest(f"Got {len(values)} weights for {donor_count} donors.")
    if any(not math.isfinite(w) or w < 0 for w in values) or sum(values) <= 0:
        raise SoupInvalidRequest(
            "Weights must be finite, non-negative and sum to a positive value."
        )
    return values


def _resolve_output(output: str | Path | None, ckpt_dir: Path) -> Path:
    if output is None:
        return ckpt_dir / DEFAULT_SOUP_FILENAME
    out = Path(output)
    # A bare file name lands next to the donors; any other path is used as given.
    if out.parent == Path("."):
        return ckpt_dir / out.name
    return out


def _default_soup_function() -> SoupFunction:
    from training.export.soup import create_pareto_model_soup

    return create_pareto_model_soup


def blend_model_soup(
    model_key: str,
    project_root: Path,
    donors: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    output: str | Path | None = None,
    soup_function: SoupFunction | None = None,
) -> SoupResult:
    """Resolve vault checkpoints and write a Pareto Model Soup checkpoint.

    The single entry point for ``lemtrain soup`` and ``POST /api/models/{key}/soup``.
    """
    paths = resolve_vault_checkpoints(model_key, project_root, donors)
    checked_weights = _validate_weights(weights, len(paths))
    out_path = _resolve_output(output, paths[0].parent)
    fn = soup_function or _default_soup_function()
    fn(paths, weights=checked_weights, output_path=out_path)
    return SoupResult(
        model_key=model_key, donors=paths, weights=checked_weights, output_path=out_path
    )
