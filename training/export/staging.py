"""Variation Staging Service for Kaggle Models Hub distributions.

Stages and strictly isolates model artifacts into canonical publishing variation folders:
1. dist/kaggle/pytorch_default/ -> checkpoints/ (*.pth) + root files (README.md, metrics.csv)
2. dist/kaggle/pytorch_fp32/    -> [ModelStem].pt (linked/copied from dist/kaggle/pytorch/)
3. dist/kaggle/onnx_fp16/       -> [ModelStem].onnx (+ [ModelStem].onnx.data above 2 GB)
4. dist/kaggle/onnx_fp32/       -> [ModelStem]_FP32.onnx + [ModelStem]_FP32.onnx.data

An ONNX file is moved together with its weights sidecar; when the stem changes the model is
re-saved so the sidecar location recorded inside the graph matches the new file name.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from training.cloud.variations import (
    ONNX_FP16_SUFFIX,
    ONNX_FP32_SUFFIX,
    PT_SUFFIX,
    PYTORCH_DEFAULT,
)
from training.export.onnx_weights import relocate_onnx, sidecar_path
from training.export.validation import (
    IsolationReport,
    StrictIsolationValidator,
    resolve_model_stem,
)
from training.registry import ModelSpecification
from training.utils.fs import remove_tree
from training.utils.paths import get_workspace_root

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
    return not path.name.endswith(ONNX_FP32_SUFFIX)


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
    """A variation folder: its main artifact, ONNX sidecar handling and purge rules."""

    folder: str
    label: str
    main: _Artifact
    sidecar_key: str | None
    """Result key of the ``.onnx.data`` weights sidecar; None for non-ONNX variations."""
    leftover_glob: str
    foreign: str
    """Log wording for files purged from the folder."""


_PT = _Artifact("pytorch", PT_SUFFIX, ("pt", "pytorch"))
_FP32 = _Artifact("onnx_fp32", ONNX_FP32_SUFFIX, ("onnx_fp32",))
_FP16 = _Artifact("onnx_fp16", ONNX_FP16_SUFFIX, ("onnx_fp16",), keep=_not_fp32)

_VARIATIONS: tuple[_Variation, ...] = (
    _Variation("pytorch", "PyTorch FP32", _PT, None, f"*{PT_SUFFIX}", "foreign file"),
    _Variation(
        "onnx_fp32", "ONNX FP32", _FP32, "onnx_fp32_data", f"*{ONNX_FP32_SUFFIX}*", "foreign file"
    ),
    _Variation(
        "onnx_fp16", "ONNX FP16", _FP16, "onnx_fp16_data", f"*{ONNX_FP16_SUFFIX}", "non-FP16 file"
    ),
)
"""Staging order: PyTorch ``.pt``, ONNX FP32 (+ sidecar), ONNX FP16 (+ sidecar above 2 GB)."""


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

    def _move_main(self, var: _Variation, src: Path, dest: Path) -> Path | None:
        """Move the main artifact; ONNX files take their sidecar along (renamed if needed)."""
        if var.sidecar_key is None:
            _transfer(src, dest)
            return None
        return relocate_onnx(src, dest)

    def stage(self, var: _Variation, out_dir: Path) -> None:
        """Move one variation into ``out_dir``, then purge foreign and leftover files."""
        main_src = self.find_source(var.main)
        dest = out_dir / f"{self.stem}{var.main.suffix}"
        dest_data = sidecar_path(dest)
        allowed = {dest.name}

        if main_src and main_src.exists():
            moved_data = self._move_main(var, main_src, dest)
            self.staged[var.main.key] = dest
            if var.sidecar_key and moved_data is not None:
                self.staged[var.sidecar_key] = moved_data
                allowed.add(moved_data.name)
            logger.info("Staged %s variation to %s", var.label, dest)
        elif dest.exists():
            self.staged[var.main.key] = dest
            if var.sidecar_key and dest_data.is_file():
                self.staged[var.sidecar_key] = dest_data
                allowed.add(dest_data.name)

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


@dataclass
class VariationStagingService:
    """Orchestrates staging of exported artifacts into strictly isolated variation directories."""

    spec: ModelSpecification | None = None
    validator: StrictIsolationValidator = field(init=False)

    def __post_init__(self) -> None:
        self.validator = StrictIsolationValidator(spec=self.spec)

    def _resolve_hub_dir(self, model_key: str, base_dir: Path | str | None = None) -> Path:
        """Resolve directory path in LemGendaryModels (default: under the workspace root)."""
        if base_dir:
            hub_dir = Path(base_dir).resolve()
        else:
            hub_dir = (get_workspace_root() / "LemGendaryModels" / model_key).resolve()
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
        stem = resolve_model_stem(model_key, self.spec)
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
