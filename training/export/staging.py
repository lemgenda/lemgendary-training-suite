"""Variation Staging Service for Kaggle Models Hub distributions.

Stages and strictly isolates model artifacts into canonical publishing variation folders:
1. dist/kaggle/pytorch_default/ -> checkpoints/ (*.pth) + root files (README.md, metrics.csv)
2. dist/kaggle/pytorch_fp32/    -> [ModelStem].pt (linked/copied from dist/kaggle/pytorch/)
3. dist/kaggle/onnx_fp16/       -> [ModelStem].onnx
4. dist/kaggle/onnx_fp32/       -> [ModelStem]_FP32.onnx + [ModelStem]_FP32.onnx.data
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from training.cloud.variations import ONNX_FP16, ONNX_FP32, PYTORCH_DEFAULT, PYTORCH_FP32, VARIATIONS
from training.export.validation import IsolationReport, StrictIsolationValidator
from training.registry import ModelSpecification, get_model_registry
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

        # Directories
        pytorch_default_dir = dist_dir / PYTORCH_DEFAULT.payload_dir
        pytorch_fp32_dir = dist_dir / "pytorch"  # Canonical storage for .pt
        onnx_fp32_dir = dist_dir / "onnx_fp32"
        onnx_fp16_dir = dist_dir / "onnx_fp16"

        pytorch_fp32_dir.mkdir(parents=True, exist_ok=True)
        onnx_fp32_dir.mkdir(parents=True, exist_ok=True)
        onnx_fp16_dir.mkdir(parents=True, exist_ok=True)

        staged: dict[str, Path] = {}
        errors: list[str] = []
        sources = dict(source_files or {})

        def _safe_copy(src: Path, dest: Path) -> None:
            if src.resolve() == dest.resolve():
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                try:
                    if src.stat().st_size == dest.stat().st_size:
                        return
                    dest.unlink()
                except OSError:
                    pass
            shutil.copy2(str(src), str(dest))

        def _transfer(src: Path, dest: Path) -> None:
            if src.resolve() == dest.resolve():
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                try:
                    if src.stat().st_size == dest.stat().st_size:
                        src.unlink()
                        return
                    dest.unlink()
                except OSError:
                    pass
            shutil.move(str(src), str(dest))

        # ---------------------------------------------------------------------
        # 1. PyTorch Default (canonical hub_dir checkpoints/ + README.md + metrics.csv)
        # ---------------------------------------------------------------------
        # The canonical location is hub_dir itself. We avoid duplicating checkpoints
        # and metrics.csv on disk to preserve single source of truth and save disk space.
        src_ckpts_dir = hub_dir / "checkpoints"
        if src_ckpts_dir.exists() and src_ckpts_dir.is_dir():
            for pth in src_ckpts_dir.glob("*.pth"):
                staged[f"default_ckpt_{pth.name}"] = pth

        for root_file_name in ("README.md", "metrics.csv"):
            rf = hub_dir / root_file_name
            if rf.exists() and rf.is_file():
                staged[f"default_{root_file_name}"] = rf

        # If a legacy pytorch_default directory exists, purge duplicate files from it
        if pytorch_default_dir.exists():
            shutil.rmtree(pytorch_default_dir, ignore_errors=True)

        # ---------------------------------------------------------------------
        # 2. Stage PyTorch FP32 (.pt)
        # ---------------------------------------------------------------------
        pt_src = sources.get("pt") or sources.get("pytorch")
        if not pt_src or not pt_src.exists():
            cands = (
                hub_dir / f"{stem}.pt",
                hub_dir / f"LemGendary{stem}.pt",
                hub_dir / f"{model_key}.pt",
            )
            for cand in cands:
                if cand.exists():
                    pt_src = cand
                    break
            if not pt_src or not pt_src.exists():
                root_pts = [f for f in hub_dir.glob("*.pt") if f.is_file()]
                if root_pts:
                    pt_src = root_pts[0]

        dest_pt = pytorch_fp32_dir / f"{stem}.pt"
        if pt_src and pt_src.exists():
            _transfer(pt_src, dest_pt)
            staged["pytorch"] = dest_pt
            logger.info("Staged PyTorch FP32 variation to %s", dest_pt)
        elif dest_pt.exists():
            staged["pytorch"] = dest_pt

        # Purge foreign files from pytorch dir and remove root leftovers
        for child in pytorch_fp32_dir.iterdir():
            if child.is_file() and child.name != f"{stem}.pt":
                child.unlink()
                logger.debug("Removed foreign file from pytorch variation: %s", child.name)

        for leftover_pt in hub_dir.glob("*.pt"):
            if leftover_pt.is_file() and dest_pt.exists():
                try:
                    leftover_pt.unlink()
                except OSError:
                    pass

        # ---------------------------------------------------------------------
        # 3. Stage ONNX FP32 (.onnx + .onnx.data)
        # ---------------------------------------------------------------------
        fp32_onnx_src = sources.get("onnx_fp32")
        if not fp32_onnx_src or not fp32_onnx_src.exists():
            cands_fp32 = (
                hub_dir / f"{stem}_FP32.onnx",
                hub_dir / f"LemGendary{stem}_FP32.onnx",
                hub_dir / f"{model_key}_FP32.onnx",
            )
            for cand in cands_fp32:
                if cand.exists():
                    fp32_onnx_src = cand
                    break
            if not fp32_onnx_src or not fp32_onnx_src.exists():
                root_fp32 = [f for f in hub_dir.glob("*_FP32.onnx") if f.is_file()]
                if root_fp32:
                    fp32_onnx_src = root_fp32[0]

        fp32_data_src = sources.get("onnx_fp32_data")
        if not fp32_data_src or not fp32_data_src.exists():
            cands_data = (
                hub_dir / f"{stem}_FP32.onnx.data",
                hub_dir / f"LemGendary{stem}_FP32.onnx.data",
                hub_dir / f"{model_key}_FP32.onnx.data",
            )
            for cand in cands_data:
                if cand.exists():
                    fp32_data_src = cand
                    break
            if not fp32_data_src or not fp32_data_src.exists():
                root_data = [f for f in hub_dir.glob("*_FP32.onnx.data") if f.is_file()]
                if root_data:
                    fp32_data_src = root_data[0]

        dest_fp32_onnx = onnx_fp32_dir / f"{stem}_FP32.onnx"
        dest_fp32_data = onnx_fp32_dir / f"{stem}_FP32.onnx.data"

        if fp32_onnx_src and fp32_onnx_src.exists():
            _transfer(fp32_onnx_src, dest_fp32_onnx)
            staged["onnx_fp32"] = dest_fp32_onnx
            if fp32_data_src and fp32_data_src.exists():
                _transfer(fp32_data_src, dest_fp32_data)
                staged["onnx_fp32_data"] = dest_fp32_data
            logger.info("Staged ONNX FP32 variation to %s", dest_fp32_onnx)
        elif dest_fp32_onnx.exists():
            staged["onnx_fp32"] = dest_fp32_onnx
            if dest_fp32_data.exists():
                staged["onnx_fp32_data"] = dest_fp32_data

        allowed_names = {f"{stem}_FP32.onnx", f"{stem}_FP32.onnx.data"}
        for child in onnx_fp32_dir.iterdir():
            if child.is_file() and child.name not in allowed_names:
                child.unlink()
                logger.debug("Removed foreign file from onnx_fp32 variation: %s", child.name)

        for leftover_fp32 in hub_dir.glob("*_FP32.onnx*"):
            if leftover_fp32.is_file() and dest_fp32_onnx.exists():
                try:
                    leftover_fp32.unlink()
                except OSError:
                    pass

        # ---------------------------------------------------------------------
        # 4. Stage ONNX FP16 (.onnx)
        # ---------------------------------------------------------------------
        fp16_src = sources.get("onnx_fp16")
        if not fp16_src or not fp16_src.exists():
            cands_fp16 = (
                hub_dir / f"{stem}.onnx",
                hub_dir / f"LemGendary{stem}.onnx",
                hub_dir / f"{model_key}.onnx",
            )
            for cand in cands_fp16:
                if cand.exists():
                    fp16_src = cand
                    break
            if not fp16_src or not fp16_src.exists():
                root_fp16 = [
                    f for f in hub_dir.glob("*.onnx")
                    if f.is_file() and not f.name.endswith("_FP32.onnx")
                ]
                if root_fp16:
                    fp16_src = root_fp16[0]

        dest_fp16 = onnx_fp16_dir / f"{stem}.onnx"
        if fp16_src and fp16_src.exists():
            _transfer(fp16_src, dest_fp16)
            staged["onnx_fp16"] = dest_fp16
            logger.info("Staged ONNX FP16 variation to %s", dest_fp16)
        elif dest_fp16.exists():
            staged["onnx_fp16"] = dest_fp16

        for child in onnx_fp16_dir.iterdir():
            if child.is_file() and child.name != f"{stem}.onnx":
                child.unlink()
                logger.debug("Removed non-FP16 file from onnx_fp16 variation: %s", child.name)

        for leftover_fp16 in hub_dir.glob("*.onnx"):
            if leftover_fp16.is_file() and not leftover_fp16.name.endswith("_FP32.onnx") and dest_fp16.exists():
                try:
                    leftover_fp16.unlink()
                except OSError:
                    pass

        # ---------------------------------------------------------------------
        # 5. Isolation Validation
        # ---------------------------------------------------------------------
        report = None
        if validate:
            report = self.validator.validate_all(model_key, hub_dir)
            if not report.is_valid:
                errors.extend(report.all_errors)

        return StagingResult(
            model_key=model_key,
            staged_files=staged,
            report=report,
            success=len(errors) == 0,
            errors=errors,
        )
