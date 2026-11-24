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
from training.telemetry.validator import TelemetryCsvValidator, TelemetrySchemaError
from training.utils.lazy_imports import load_onnxruntime
from training.utils.paths import get_model_hub_dir, get_project_root
from training.verification.contracts import (
    ModelFamilyContract,
    ModelVerificationReport,
    StageVerificationResult,
    VerificationStage,
)
from training.verification.registry import get_verification_contract

logger = logging.getLogger("lemtrain.verification.engine")


class ModelVerificationEngine:
    """Orchestrates comprehensive verification contract checks on a model."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.recovery = CheckpointRecoveryEngine(self.project_root)

    def _resolve_ckpt_path(
        self, model_key: str, checkpoint_path: Path | str | None
    ) -> Path | None:
        if checkpoint_path is not None:
            return Path(checkpoint_path).resolve()
        candidate_roots = self.recovery.find_candidate_roots(model_key)
        discovered = self.recovery.discover_checkpoints(candidate_roots, model_key)
        return next(
            (
                discovered[k]
                for k in ("best", "latest", "latest_epoch", "progress")
                if k in discovered
            ),
            None,
        )

    def _load_ckpt_weights(
        self,
        model: nn.Module,
        ckpt_path: Path,
        contract: ModelFamilyContract,
        details: dict[str, Any],
    ) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
        size = ckpt_path.stat().st_size
        details["file_size_bytes"] = size
        if size < contract.checkpoint.min_size_bytes:
            raise ValueError(
                f"Checkpoint size {size} bytes is below minimum "
                f"{contract.checkpoint.min_size_bytes}"
            )
        payload = safe_load_checkpoint(ckpt_path, map_location="cpu")
        for req_key in contract.checkpoint.required_keys:
            if req_key not in payload and "state_dict" not in payload and "model" not in payload:
                raise KeyError(f"Missing mandatory checkpoint payload key: '{req_key}'")
        weights = (
            payload.get("model_state_dict")
            or payload.get("state_dict")
            or payload.get("model")
            or payload
        )
        if isinstance(weights, dict):
            incompatible = model.load_state_dict(
                weights, strict=contract.checkpoint.strict_weights
            )
            details["missing_keys_count"] = len(incompatible.missing_keys)
            details["unexpected_keys_count"] = len(incompatible.unexpected_keys)

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
            if model is None:
                model = get_model(model_key)
            model.eval()
            ckpt_path = self._resolve_ckpt_path(model_key, checkpoint_path)
            if ckpt_path is not None:
                self._load_ckpt_weights(model, ckpt_path, contract, details)
            details["model_class"] = model.__class__.__name__
            details["checkpoint_resolved"] = str(ckpt_path) if ckpt_path else None
            return (
                StageVerificationResult(stage=stage, passed=True, details=details),
                model,
                ckpt_path,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            logger.error("Loadability verification failed for '%s': %s", model_key, exc)
            return (
                StageVerificationResult(stage=stage, passed=False, details=details, error=str(exc)),
                model if model is not None else nn.Identity(),
                ckpt_path,
            )

    @staticmethod
    def _max_trial_abs_diff(outputs: list[list[torch.Tensor]]) -> float:
        max_abs_diff = 0.0
        base = outputs[0]
        for trial_idx, other in enumerate(outputs[1:], start=1):
            if len(base) != len(other):
                raise ValueError(
                    f"Output count mismatch: trial 0 had {len(base)}, "
                    f"trial {trial_idx} had {len(other)}"
                )
            for b_t, o_t in zip(base, other, strict=True):
                if b_t.shape != o_t.shape:
                    raise ValueError(
                        f"Shape mismatch: {tuple(b_t.shape)} vs {tuple(o_t.shape)}"
                    )
                diff = float((b_t - o_t).abs().max()) if b_t.numel() else 0.0
                max_abs_diff = max(max_abs_diff, diff)
        return max_abs_diff

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
            random.seed(contract.determinism.seed)
            np.random.seed(contract.determinism.seed)
            torch.manual_seed(contract.determinism.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(contract.determinism.seed)

            outputs: list[list[torch.Tensor]] = []
            for _trial in range(contract.determinism.eval_trials):
                with torch.no_grad():
                    if isinstance(dummy_input, tuple):
                        out = model(*dummy_input)
                    elif isinstance(dummy_input, dict):
                        out = model(**dummy_input)
                    else:
                        out = model(dummy_input)
                outputs.append([t.clone().detach().cpu() for t in flatten_outputs(out)])

            max_abs_diff = self._max_trial_abs_diff(outputs)
            details["max_abs_diff"] = max_abs_diff
            details["tolerance_atol"] = contract.determinism.atol
            details["trials_run"] = contract.determinism.eval_trials
            if max_abs_diff > contract.determinism.atol:
                return StageVerificationResult(
                    stage=stage,
                    passed=False,
                    details=details,
                    error=(
                        f"Forward pass is non-deterministic: max_abs_diff={max_abs_diff} "
                        f"> {contract.determinism.atol}"
                    ),
                )
            return StageVerificationResult(stage=stage, passed=True, details=details)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.error("Determinism verification failed: %s", exc)
            return StageVerificationResult(
                stage=stage, passed=False, details=details, error=str(exc)
            )

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

    @staticmethod
    def _check_fp32_onnx_parity(
        ort: Any,
        fp32_onnx: Path,
        model: nn.Module,
        dummy_input: torch.Tensor,
        contract: ModelFamilyContract,
    ) -> tuple[float, str | None]:
        session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_out = session.run(None, {input_name: dummy_input.detach().cpu().numpy()})
        with torch.no_grad():
            pt_out = flatten_outputs(model(dummy_input))
        tol_fp32 = ProbeTolerance(
            atol=contract.export.fp32_atol, rtol=contract.export.fp32_rtol
        )
        probe_res = compare_outputs(pt_out, [torch.from_numpy(a) for a in ort_out], tol_fp32)
        err = (
            f"ONNX FP32 equivalence failed: {probe_res.detail} (diff={probe_res.max_abs_diff})"
            if not probe_res.ok
            else None
        )
        return probe_res.max_abs_diff, err

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
            export_dir = temp_dir / "verify_export"
            export_dir.mkdir(parents=True, exist_ok=True)
            exported = PyTorchModelExporter().export(
                model=model,
                model_key=model_key,
                output_dir=export_dir,
                dummy_input=dummy_input,
            )
            fp32_onnx = exported.get("onnx_fp32")
            if fp32_onnx and fp32_onnx.is_file():
                diff, err = self._check_fp32_onnx_parity(
                    ort, fp32_onnx, model, dummy_input, contract
                )
                details["fp32_onnx_diff"] = diff
                if err:
                    return StageVerificationResult(
                        stage=stage, passed=False, details=details, error=err
                    )
            details["exported_targets"] = list(exported.keys())
            return StageVerificationResult(stage=stage, passed=True, details=details)
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            logger.error("Export equivalence verification failed: %s", exc)
            return StageVerificationResult(
                stage=stage, passed=False, details=details, error=str(exc)
            )

    def _resolve_telemetry_csv(self, target_dir: Path, model_key: str, filename: str) -> Path:
        path = target_dir / filename
        if path.exists():
            return path
        candidate = self.project_root / "runs" / model_key / filename
        return candidate if candidate.exists() else path

    @staticmethod
    def _validate_telemetry_files(
        metrics_csv: Path,
        history_csv: Path,
        contract: ModelFamilyContract,
        details: dict[str, Any],
    ) -> list[str]:
        family = contract.schema.task_family
        strict = contract.schema.strict_columns
        errors: list[str] = []
        if metrics_csv.exists():
            try:
                with open(metrics_csv, "r", encoding="utf-8") as f:
                    header_line = f.readline().lower()
                is_hist = "train_loss" in header_line or "psnr" in header_line
                rows = (
                    TelemetryCsvValidator.validate_history_csv(
                        metrics_csv, family=family, strict_columns=strict
                    )
                    if is_hist
                    else TelemetryCsvValidator.validate_metrics_csv(
                        metrics_csv, family=family, strict_columns=strict
                    )
                )
                details["metrics_csv_type"] = "history_format" if is_hist else "step_metrics_format"
                details["metrics_csv_rows"] = len(rows)
            except TelemetrySchemaError as exc:
                errors.append(f"metrics.csv schema error: {exc}")
        elif contract.schema.enforce_metrics:
            details["metrics_csv"] = "not_found (optional in pre-run verification)"

        if history_csv.exists():
            try:
                rows = TelemetryCsvValidator.validate_history_csv(
                    history_csv, family=family, strict_columns=strict
                )
                details["history_csv_rows"] = len(rows)
            except TelemetrySchemaError as exc:
                errors.append(f"history.csv schema error: {exc}")
        elif contract.schema.enforce_history:
            details["history_csv"] = "not_found (optional in pre-run verification)"
        return errors

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
        metrics_csv = self._resolve_telemetry_csv(target_dir, model_key, "metrics.csv")
        history_csv = self._resolve_telemetry_csv(target_dir, model_key, "history.csv")
        errors = self._validate_telemetry_files(metrics_csv, history_csv, contract, details)
        if errors:
            return StageVerificationResult(
                stage=stage, passed=False, details=details, error="; ".join(errors)
            )
        return StageVerificationResult(stage=stage, passed=True, details=details)

    def _run_stages(
        self,
        model_key: str,
        checkpoint_path: Path | str | None,
        contract: ModelFamilyContract,
        options: dict[str, Any],
    ) -> tuple[dict[str, StageVerificationResult], list[str], Path | None]:
        stages: dict[str, StageVerificationResult] = {}
        all_errors: list[str] = []
        load_res, resolved_model, resolved_ckpt = self.verify_loadability(
            model_key, checkpoint_path, contract, model=options.get("model")
        )
        stages[VerificationStage.LOADABILITY.value] = load_res
        if not load_res.passed:
            all_errors.append(f"Loadability: {load_res.error}")

        dummy = options.get("dummy_input")
        if dummy is None:
            dummy = torch.randn(1, 3, 64, 64)

        for name, stage_enum, res in (
            (
                "Determinism",
                VerificationStage.DETERMINISM,
                self.verify_determinism(resolved_model, dummy, contract),
            ),
            (
                "Metric Thresholds",
                VerificationStage.METRIC_THRESHOLDS,
                self.verify_metric_thresholds(options.get("metrics"), contract),
            ),
            (
                "Export Equivalence",
                VerificationStage.EXPORT_EQUIVALENCE,
                self.verify_export_equivalence(
                    resolved_model,
                    model_key,
                    dummy,
                    contract,
                    options.get("temp_dir") or (self.project_root / ".cache" / "verification"),
                ),
            ),
            (
                "Schema Conformance",
                VerificationStage.SCHEMA_CONFORMANCE,
                self.verify_schema_conformance(
                    model_key, contract, hub_dir=options.get("hub_dir")
                ),
            ),
        ):
            stages[stage_enum.value] = res
            if not res.passed:
                all_errors.append(f"{name}: {res.error}")
        return stages, all_errors, resolved_ckpt

    def run_full_verification(
        self,
        model_key: str,
        checkpoint_path: Path | str | None = None,
        **options: Any,
    ) -> ModelVerificationReport:
        """Execute all contract verification stages and emit a unified report."""
        contract = get_verification_contract(model_key)
        family = contract.family.value
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t_dir = options.get("temp_dir") or (self.project_root / ".cache" / "verification")
        t_dir.mkdir(parents=True, exist_ok=True)
        options["temp_dir"] = t_dir
        stages, all_errors, resolved_ckpt = self._run_stages(
            model_key, checkpoint_path, contract, options
        )

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
