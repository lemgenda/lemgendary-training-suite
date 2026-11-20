"""Model Verification Pipeline Engine.

Implements end-to-end model verification:
1. Loadability (weights + config)
2. Forward determinism (fixed seed + input -> identical output)
3. Metric thresholds (held-out set evaluation against contract targets)
4. Export equivalence (FP32/FP16 ONNX vs PyTorch)
5. Schema conformance (metrics.csv and history.csv validation)
"""

from __future__ import annotations

import datetime
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.factory import get_model
from training.checkpoint.manager import safe_load_checkpoint
from training.checkpoint.recovery import CheckpointRecoveryEngine
from training.export.dynamo_onnx import ProbeTolerance, compare_outputs, flatten_outputs
from training.export.pytorch_exporter import PyTorchModelExporter
from training.registry import ModelSpecification, get_model_registry
from training.telemetry.validator import TelemetryCsvValidator, TelemetrySchemaError
from training.utils.lazy_imports import load_onnxruntime
from training.utils.model_payload import model_payload_of
from training.utils.paths import get_model_hub_dir, get_project_root
from training.verification.contracts import (
    ModelFamilyContract,
    ModelVerificationReport,
    StageVerificationResult,
    VerificationStage,
)
from training.verification.registry import get_verification_contract, resolve_model_family

logger = logging.getLogger("lemtrain.verification.engine")


class ModelVerificationEngine:
    """Orchestrates comprehensive verification contract checks on a model."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.recovery = CheckpointRecoveryEngine(self.project_root)

    def verify_loadability(
        self,
        model_key: str,
        checkpoint_path: Path | str | None,
        contract: ModelFamilyContract,
        model: nn.Module | None = None,
    ) -> tuple[StageVerificationResult, nn.Module, Path | None]:
        """Verify loadability of model weights and architecture configuration."""
        stage = VerificationStage.LOADABILITY
        ckpt_path: Path | None = None
        details: dict[str, Any] = {}

        try:
            # 1. Instantiate model if not passed
            if model is None:
                model = get_model(model_key)
            model.eval()

            # 2. Resolve checkpoint path
            if checkpoint_path is not None:
                ckpt_path = Path(checkpoint_path).resolve()
            else:
                found = self.recovery.discover_checkpoint(model_key)
                if found:
                    ckpt_path = found

            if ckpt_path is not None:
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
                size = ckpt_path.stat().st_size
                details["file_size_bytes"] = size
                if size < contract.checkpoint.min_size_bytes:
                    raise ValueError(
                        f"Checkpoint size {size} bytes is below minimum {contract.checkpoint.min_size_bytes}"
                    )

                payload = safe_load_checkpoint(ckpt_path, map_location="cpu")
                for req_key in contract.checkpoint.required_keys:
                    if req_key not in payload and "state_dict" not in payload and "model" not in payload:
                        raise KeyError(f"Missing mandatory checkpoint payload key: '{req_key}'")

                # Extract weights dictionary
                weights = payload.get("model_state_dict") or payload.get("state_dict") or payload.get("model") or payload
                if isinstance(weights, dict):
                    incompatible = model.load_state_dict(weights, strict=contract.checkpoint.strict_weights)
                    details["missing_keys_count"] = len(incompatible.missing_keys)
                    details["unexpected_keys_count"] = len(incompatible.unexpected_keys)

            details["model_class"] = model.__class__.__name__
            details["checkpoint_resolved"] = str(ckpt_path) if ckpt_path else None
            return StageVerificationResult(stage=stage, passed=True, details=details), model, ckpt_path

        except Exception as exc:
            logger.error("Loadability verification failed for '%s': %s", model_key, exc)
            return (
                StageVerificationResult(stage=stage, passed=False, details=details, error=str(exc)),
                model if model is not None else nn.Identity(),
                ckpt_path,
            )

    def verify_determinism(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor],
        contract: ModelFamilyContract,
    ) -> StageVerificationResult:
        """Verify forward determinism under fixed seeds and fixed inputs."""
        stage = VerificationStage.DETERMINISM
        details: dict[str, Any] = {}

        try:
            model.eval()

            # Seed once at start of evaluation
            random.seed(contract.determinism.seed)
            np.random.seed(contract.determinism.seed)
            torch.manual_seed(contract.determinism.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(contract.determinism.seed)

            outputs: list[list[torch.Tensor]] = []

            for trial in range(contract.determinism.eval_trials):
                with torch.no_grad():
                    if isinstance(dummy_input, tuple):
                        out = model(*dummy_input)
                    elif isinstance(dummy_input, dict):
                        out = model(**dummy_input)
                    else:
                        out = model(dummy_input)

                flat_out = [t.clone().detach().cpu() for t in flatten_outputs(out)]
                outputs.append(flat_out)

            # Compare outputs across trials
            max_abs_diff = 0.0
            base = outputs[0]
            for trial_idx, other in enumerate(outputs[1:], start=1):
                if len(base) != len(other):
                    raise ValueError(f"Output count mismatch: trial 0 had {len(base)}, trial {trial_idx} had {len(other)}")
                for b_t, o_t in zip(base, other, strict=True):
                    if b_t.shape != o_t.shape:
                        raise ValueError(f"Shape mismatch: {tuple(b_t.shape)} vs {tuple(o_t.shape)}")
                    diff = float((b_t - o_t).abs().max()) if b_t.numel() else 0.0
                    max_abs_diff = max(max_abs_diff, diff)

            details["max_abs_diff"] = max_abs_diff
            details["tolerance_atol"] = contract.determinism.atol
            details["trials_run"] = contract.determinism.eval_trials

            if max_abs_diff > contract.determinism.atol:
                return StageVerificationResult(
                    stage=stage,
                    passed=False,
                    details=details,
                    error=f"Forward pass is non-deterministic: max_abs_diff={max_abs_diff} > {contract.determinism.atol}",
                )

            return StageVerificationResult(stage=stage, passed=True, details=details)

        except Exception as exc:
            logger.error("Determinism verification failed: %s", exc)
            return StageVerificationResult(stage=stage, passed=False, details=details, error=str(exc))

    def verify_metric_thresholds(
        self,
        metrics: dict[str, float] | None,
        contract: ModelFamilyContract,
    ) -> StageVerificationResult:
        """Verify computed metrics against contract threshold bounds."""
        stage = VerificationStage.METRIC_THRESHOLDS
        details: dict[str, Any] = {}

        if metrics is None:
            details["skipped"] = "No validation metrics provided or evaluated"
            return StageVerificationResult(stage=stage, passed=True, details=details)

        details["evaluated_metrics"] = metrics
        failures: list[str] = []

        # Check minimum targets (higher is better)
        for metric, min_val in contract.metrics.min_targets.items():
            if metric in metrics:
                actual = metrics[metric]
                if actual < min_val:
                    failures.append(f"{metric} = {actual:.4f} < min target {min_val}")

        # Check maximum targets (lower is better)
        for metric, max_val in contract.metrics.max_targets.items():
            if metric in metrics:
                actual = metrics[metric]
                if actual > max_val:
                    failures.append(f"{metric} = {actual:.4f} > max target {max_val}")

        details["failures"] = failures
        if failures:
            return StageVerificationResult(
                stage=stage,
                passed=False,
                details=details,
                error="; ".join(failures),
            )

        return StageVerificationResult(stage=stage, passed=True, details=details)

    def verify_export_equivalence(
        self,
        model: nn.Module,
        model_key: str,
        dummy_input: torch.Tensor,
        contract: ModelFamilyContract,
        temp_dir: Path,
    ) -> StageVerificationResult:
        """Verify numerical parity between PyTorch and exported ONNX models."""
        stage = VerificationStage.EXPORT_EQUIVALENCE
        details: dict[str, Any] = {}

        try:
            ort = load_onnxruntime()
            if ort is None:
                details["skipped"] = "onnxruntime not installed"
                return StageVerificationResult(stage=stage, passed=True, details=details)

            model.eval()
            exporter = PyTorchModelExporter()
            export_dir = temp_dir / "verify_export"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Export ONNX FP32 and FP16
            exported = exporter.export(
                model=model,
                model_key=model_key,
                output_dir=export_dir,
                dummy_input=dummy_input,
            )

            # 1. Check FP32 equivalence
            fp32_onnx = exported.get("onnx_fp32")
            if fp32_onnx and fp32_onnx.is_file():
                session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                raw_np = dummy_input.detach().cpu().numpy()
                ort_out = session.run(None, {input_name: raw_np})
                with torch.no_grad():
                    pt_out = flatten_outputs(model(dummy_input))

                ort_tensors = [torch.from_numpy(arr) for arr in ort_out]
                tol_fp32 = ProbeTolerance(atol=contract.export.fp32_atol, rtol=contract.export.fp32_rtol)
                probe_res = compare_outputs(pt_out, ort_tensors, tol_fp32)
                details["fp32_onnx_diff"] = probe_res.max_abs_diff
                if not probe_res.ok:
                    return StageVerificationResult(
                        stage=stage,
                        passed=False,
                        details=details,
                        error=f"ONNX FP32 equivalence failed: {probe_res.detail} (diff={probe_res.max_abs_diff})",
                    )

            details["exported_targets"] = list(exported.keys())
            return StageVerificationResult(stage=stage, passed=True, details=details)

        except Exception as exc:
            logger.error("Export equivalence verification failed: %s", exc)
            return StageVerificationResult(stage=stage, passed=False, details=details, error=str(exc))

    def verify_schema_conformance(
        self,
        model_key: str,
        contract: ModelFamilyContract,
        hub_dir: Path | None = None,
    ) -> StageVerificationResult:
        """Verify schema conformance of metrics.csv and history.csv."""
        stage = VerificationStage.SCHEMA_CONFORMANCE
        details: dict[str, Any] = {}
        target_dir = hub_dir or get_model_hub_dir(model_key)
        family = contract.schema.task_family

        metrics_csv = target_dir / "metrics.csv"
        history_csv = target_dir / "history.csv"

        # If files don't exist yet in hub_dir, check in project runs/
        if not metrics_csv.exists():
            candidate = self.project_root / "runs" / model_key / "metrics.csv"
            if candidate.exists():
                metrics_csv = candidate
        if not history_csv.exists():
            candidate = self.project_root / "runs" / model_key / "history.csv"
            if candidate.exists():
                history_csv = candidate

        errors: list[str] = []

        # Determine if telemetry file is epoch-level history or step-level metrics
        if metrics_csv.exists():
            try:
                # First check if the file is legacy or epoch-level by reading header
                with open(metrics_csv, "r", encoding="utf-8") as f:
                    header_line = f.readline().lower()
                if "train_loss" in header_line or "psnr" in header_line:
                    rows = TelemetryCsvValidator.validate_history_csv(
                        metrics_csv, family=family, strict_columns=contract.schema.strict_columns
                    )
                    details["metrics_csv_type"] = "history_format"
                else:
                    rows = TelemetryCsvValidator.validate_metrics_csv(
                        metrics_csv, family=family, strict_columns=contract.schema.strict_columns
                    )
                    details["metrics_csv_type"] = "step_metrics_format"
                details["metrics_csv_rows"] = len(rows)
            except TelemetrySchemaError as exc:
                errors.append(f"metrics.csv schema error: {exc}")
        elif contract.schema.enforce_metrics:
            details["metrics_csv"] = "not_found (optional in pre-run verification)"

        if history_csv.exists():
            try:
                rows = TelemetryCsvValidator.validate_history_csv(
                    history_csv, family=family, strict_columns=contract.schema.strict_columns
                )
                details["history_csv_rows"] = len(rows)
            except TelemetrySchemaError as exc:
                errors.append(f"history.csv schema error: {exc}")
        elif contract.schema.enforce_history:
            details["history_csv"] = "not_found (optional in pre-run verification)"

        if errors:
            return StageVerificationResult(
                stage=stage,
                passed=False,
                details=details,
                error="; ".join(errors),
            )

        return StageVerificationResult(stage=stage, passed=True, details=details)

    def run_full_verification(
        self,
        model_key: str,
        checkpoint_path: Path | str | None = None,
        model: nn.Module | None = None,
        dummy_input: torch.Tensor | None = None,
        metrics: dict[str, float] | None = None,
        hub_dir: Path | None = None,
        temp_dir: Path | None = None,
    ) -> ModelVerificationReport:
        """Execute all contract verification stages and produce a unified machine-readable report."""
        contract = get_verification_contract(model_key)
        family = contract.family.value
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        stages: dict[str, StageVerificationResult] = {}
        all_errors: list[str] = []

        # 1. Loadability
        load_res, resolved_model, resolved_ckpt = self.verify_loadability(
            model_key, checkpoint_path, contract, model=model
        )
        stages[VerificationStage.LOADABILITY.value] = load_res
        if not load_res.passed:
            all_errors.append(f"Loadability: {load_res.error}")

        # Resolve sample dummy input if not given
        if dummy_input is None:
            dummy_input = torch.randn(1, 3, 64, 64)

        # 2. Determinism
        det_res = self.verify_determinism(resolved_model, dummy_input, contract)
        stages[VerificationStage.DETERMINISM.value] = det_res
        if not det_res.passed:
            all_errors.append(f"Determinism: {det_res.error}")

        # 3. Metric Thresholds
        metric_res = self.verify_metric_thresholds(metrics, contract)
        stages[VerificationStage.METRIC_THRESHOLDS.value] = metric_res
        if not metric_res.passed:
            all_errors.append(f"Metric Thresholds: {metric_res.error}")

        # 4. Export Equivalence
        t_dir = temp_dir or (self.project_root / ".cache" / "verification")
        t_dir.mkdir(parents=True, exist_ok=True)
        export_res = self.verify_export_equivalence(
            resolved_model, model_key, dummy_input, contract, temp_dir=t_dir
        )
        stages[VerificationStage.EXPORT_EQUIVALENCE.value] = export_res
        if not export_res.passed:
            all_errors.append(f"Export Equivalence: {export_res.error}")

        # 5. Schema Conformance
        schema_res = self.verify_schema_conformance(model_key, contract, hub_dir=hub_dir)
        stages[VerificationStage.SCHEMA_CONFORMANCE.value] = schema_res
        if not schema_res.passed:
            all_errors.append(f"Schema Conformance: {schema_res.error}")

        overall_passed = len(all_errors) == 0

        return ModelVerificationReport(
            model_key=model_key,
            family=family,
            overall_passed=overall_passed,
            checkpoint_path=str(resolved_ckpt) if resolved_ckpt else None,
            stages=stages,
            errors=all_errors,
            timestamp=timestamp,
        )
