"""Variation Staging Service for Kaggle Models Hub distributions.

Stages and strictly isolates model artifacts into canonical publishing variation folders:
1. dist/kaggle/pytorch_default/ -> checkpoints/ (*.pth) + root files (README.md, metrics.csv)
2. dist/kaggle/pytorch_fp32/    -> [ModelStem].pt (linked/copied from dist/kaggle/pytorch/)
3. dist/kaggle/onnx_fp16/       -> [ModelStem].onnx
4. dist/kaggle/onnx_fp32/       -> [ModelStem]_FP32.onnx + [ModelStem]_FP32.onnx.data
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from training.cloud.variations import (
    PYTORCH_DEFAULT,
)
from training.export.validation import IsolationReport, StrictIsolationValidator
from training.registry import ModelSpecification, get_model_registry
from training.utils.fs import remove_tree
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.staging")


@dataclass
class StagingResult:
    """Outcome of model variation staging and isolation validation."""

    model_key: str
    staged_files: dict[str, Path] = field(default_factory=dict)
    report: IsolationReport | None = None
    success: bool = True
    errors: list[str] = field(default_factory=list)


def _transfer(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest``; an equal-size ``dest`` is kept and ``src`` deleted."""
    if src.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if src.stat().st_size == dest.stat().st_size:
                src.unlink()
                return
            dest.unlink()
        except OSError as pre_err:
            logger.debug("Could not compare/replace existing %s before move: %s", dest, pre_err)
    shutil.move(str(src), str(dest))


def _not_fp32(path: Path) -> bool:
    return not path.name.endswith("_FP32.onnx")


def _any_file(_path: Path) -> bool:
    return True


@dataclass(frozen=True)
class _Artifact:
    """One staged file kind: ``<stem><suffix>`` inside a ``dist/kaggle`` folder."""

    key: str
    suffix: str
    source_keys: tuple[str, ...]
    keep: Callable[[Path], bool] = _any_file
    """Filter for the hub-root glob fallback (FP16 skips ``*_FP32.onnx``)."""


@dataclass(frozen=True)
class _Variation:
    """A variation folder: its main artifact, an optional companion and purge rules."""

    folder: str
    label: str
    main: _Artifact
    companion: _Artifact | None
    leftover_glob: str
    foreign: str
    """Log wording for files purged from the folder."""


_PT = _Artifact("pytorch", ".pt", ("pt", "pytorch"))
_FP32 = _Artifact("onnx_fp32", "_FP32.onnx", ("onnx_fp32",))
_FP32_DATA = _Artifact("onnx_fp32_data", "_FP32.onnx.data", ("onnx_fp32_data",))
_FP16 = _Artifact("onnx_fp16", ".onnx", ("onnx_fp16",), keep=_not_fp32)

_VARIATIONS: tuple[_Variation, ...] = (
    _Variation("pytorch", "PyTorch FP32", _PT, None, "*.pt", "foreign file"),
    _Variation("onnx_fp32", "ONNX FP32", _FP32, _FP32_DATA, "*_FP32.onnx*", "foreign file"),
    _Variation("onnx_fp16", "ONNX FP16", _FP16, None, "*.onnx", "non-FP16 file"),
)
"""Staging order: PyTorch ``.pt``, ONNX FP32 (+ external data), ONNX FP16."""


@dataclass
class _StagingRun:
    """State of one ``stage_variations`` call."""

    model_key: str
    hub_dir: Path
    stem: str
    sources: dict[str, Path]
    staged: dict[str, Path] = field(default_factory=dict)

    def stage_default(self, dist_dir: Path) -> None:
        """Record the canonical hub files (checkpoints, README, metrics) in place."""
        # The canonical location is hub_dir itself. We avoid duplicating checkpoints
        # and metrics.csv on disk to preserve single source of truth and save disk space.
        src_ckpts_dir = self.hub_dir / "checkpoints"
        if src_ckpts_dir.exists() and src_ckpts_dir.is_dir():
            for pth in src_ckpts_dir.glob("*.pth"):
                self.staged[f"default_ckpt_{pth.name}"] = pth

        for root_file_name in ("README.md", "metrics.csv"):
            rf = self.hub_dir / root_file_name
            if rf.exists() and rf.is_file():
                self.staged[f"default_{root_file_name}"] = rf

        # If a legacy pytorch_default directory exists, purge duplicate files from it
        pytorch_default_dir = dist_dir / PYTORCH_DEFAULT.payload_dir
        if pytorch_default_dir.exists():
            remove_tree(pytorch_default_dir, purpose="legacy pytorch_default purge")

    def find_source(self, art: _Artifact) -> Path | None:
        """Explicit source, else a known hub-root name, else the first hub-root match."""
        src: Path | None = None
        for key in art.source_keys:
            src = src or self.sources.get(key)
        if src and src.exists():
            return src
        names = (f"{self.stem}", f"LemGendary{self.stem}", f"{self.model_key}")
        for name in names:
            cand = self.hub_dir / f"{name}{art.suffix}"
            if cand.exists():
                return cand
        root = [f for f in self.hub_dir.glob(f"*{art.suffix}") if f.is_file() and art.keep(f)]
        return root[0] if root else src

    def stage(self, var: _Variation, out_dir: Path) -> None:
        """Move one variation into ``out_dir``, then purge foreign and leftover files."""
        main_src = self.find_source(var.main)
        companion_src = self.find_source(var.companion) if var.companion else None
        dest = out_dir / f"{self.stem}{var.main.suffix}"
        allowed = {dest.name}
        companion_dest = None
        if var.companion:
            companion_dest = out_dir / f"{self.stem}{var.companion.suffix}"
            allowed.add(companion_dest.name)

        if main_src and main_src.exists():
            _transfer(main_src, dest)
            self.staged[var.main.key] = dest
            if var.companion and companion_dest and companion_src and companion_src.exists():
                _transfer(companion_src, companion_dest)
                self.staged[var.companion.key] = companion_dest
            logger.info("Staged %s variation to %s", var.label, dest)
        elif dest.exists():
            self.staged[var.main.key] = dest
            if var.companion and companion_dest and companion_dest.exists():
                self.staged[var.companion.key] = companion_dest

        for child in out_dir.iterdir():
            if child.is_file() and child.name not in allowed:
                child.unlink()
                logger.debug(
                    "Removed %s from %s variation: %s", var.foreign, var.folder, child.name
                )

        for leftover in self.hub_dir.glob(var.leftover_glob):
            if leftover.is_file() and var.main.keep(leftover) and dest.exists():
                try:
                    leftover.unlink()
                except OSError as unlink_err:
                    logger.warning("Could not remove leftover %s: %s", leftover, unlink_err)


class VariationStagingService:
    """Orchestrates staging of exported artifacts into strictly isolated variation directories."""

    def __init__(self, spec: ModelSpecification | None = None) -> None:
        self.spec = spec
        self.validator = StrictIsolationValidator(spec=spec)

    def _resolve_stem(self, model_key: str) -> str:
        """Resolve canonical model binary stem from specification."""
        if self.spec and self.spec.filename:
            return self.spec.filename
        reg = get_model_registry()
        spec = reg.get_model(model_key)
        if spec and spec.filename:
            return spec.filename
        return model_key

    def _resolve_hub_dir(self, model_key: str, base_dir: Path | str | None = None) -> Path:
        """Resolve directory path in LemGendaryModels."""
        if base_dir:
            hub_dir = Path(base_dir).resolve()
        else:
            root = get_project_root()
            hub_dir = (root.parent / "LemGendaryModels" / model_key).resolve()
        hub_dir.mkdir(parents=True, exist_ok=True)
        return hub_dir

    def stage_variations(
        self,
        model_key: str,
        source_files: dict[str, Path] | None = None,
        base_dir: Path | str | None = None,
        validate: bool = True,
    ) -> StagingResult:
        """Stage all 4 model variations into dist/kaggle/ subdirectories."""
        hub_dir = self._resolve_hub_dir(model_key, base_dir)
        stem = self._resolve_stem(model_key)
        dist_dir = hub_dir / "dist" / "kaggle"
        out_dirs = [dist_dir / var.folder for var in _VARIATIONS]
        for out_dir in out_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)

        run = _StagingRun(model_key, hub_dir, stem, dict(source_files or {}))
        # 1. PyTorch Default (canonical hub_dir checkpoints/ + README.md + metrics.csv)
        run.stage_default(dist_dir)
        # 2-4. PyTorch FP32 (.pt), ONNX FP32 (.onnx + .onnx.data), ONNX FP16 (.onnx)
        for var, out_dir in zip(_VARIATIONS, out_dirs, strict=True):
            run.stage(var, out_dir)

        # 5. Isolation Validation
        errors: list[str] = []
        report = None
        if validate:
            report = self.validator.validate_all(model_key, hub_dir)
            if not report.is_valid:
                errors.extend(report.all_errors)

        return StagingResult(
            model_key=model_key,
            staged_files=run.staged,
            report=report,
            success=len(errors) == 0,
            errors=errors,
        )
