"""Fail-fast environment preflight validation engine for training suite.

Validates hardware (GPU/CUDA/MPS/CPU, VRAM headroom, single vs multi-GPU),
disk space headroom, required python package dependencies, and dataset presence
across Local, Kaggle Cloud, and Google Colab environments.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from training.hardware.discovery import DeviceInfo, discover_device

logger = logging.getLogger("lemtrain.environment.preflight")


@dataclass(frozen=True)
class PreflightIssue:
    """A detected anomaly or critical failure in environment readiness."""

    category: str
    message: str
    critical: bool = True
    remediation: str = ""


@dataclass(frozen=True)
class PreflightReport:
    """Consolidated summary of environment preflight checks."""

    env: str
    passed: bool
    issues: tuple[PreflightIssue, ...] = field(default_factory=tuple)
    device_info: DeviceInfo | None = None
    disk_free_gb: float = 0.0

    @property
    def has_critical_failures(self) -> bool:
        """Whether any critical issue blocked preflight."""
        return any(issue.critical for issue in self.issues)


class EnvironmentPreflight:
    """Fail-fast validator executing Step 0 environment sanity inspections."""

    def __init__(
        self,
        env: str = "local",
        min_disk_free_gb: float = 2.0,
        required_packages: tuple[str, ...] = ("torch", "yaml"),
    ) -> None:
        self.env = env
        self.min_disk_free_gb = min_disk_free_gb
        self.required_packages = required_packages

    def check_packages(self) -> list[PreflightIssue]:
        """Verify presence of core platform packages."""
        issues: list[PreflightIssue] = []
        for pkg in self.required_packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                issues.append(
                    PreflightIssue(
                        category="packages",
                        message=f"Required package '{pkg}' is missing from environment.",
                        critical=True,
                        remediation=f"Run: pip install {pkg}",
                    )
                )
        return issues

    def check_disk_space(self, target_dir: Path | str) -> tuple[float, list[PreflightIssue]]:
        """Verify sufficient working storage space for checkpoints and logs."""
        issues: list[PreflightIssue] = []
        target = Path(target_dir).resolve()
        # Find nearest existing directory ancestor
        check_path = target
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent

        try:
            usage = shutil.disk_usage(check_path)
            free_gb = usage.free / (1024**3)
        except OSError as exc:
            logger.warning("Could not determine disk usage at %s: %s", check_path, exc)
            return 0.0, issues

        if free_gb < self.min_disk_free_gb:
            issues.append(
                PreflightIssue(
                    category="disk",
                    message=(
                        f"Free disk space ({free_gb:.1f} GB) is below safety "
                        f"threshold ({self.min_disk_free_gb:.1f} GB)."
                    ),
                    critical=True,
                    remediation=(
                        "Purge temporary files, prune stale checkpoints, or attach storage."
                    ),
                )
            )
        return free_gb, issues

    def check_hardware(self, model_key: str = "") -> tuple[DeviceInfo | None, list[PreflightIssue]]:
        """Audit hardware accelerator readiness, VRAM limits, and GPU affinity."""
        issues: list[PreflightIssue] = []
        try:
            dev_info = discover_device()
        except RuntimeError as exc:
            issues.append(
                PreflightIssue(
                    category="hardware",
                    message=f"Hardware device discovery failed: {exc}",
                    critical=True,
                    remediation="Check CUDA driver installation or PyTorch CUDA compatibility.",
                )
            )
            return None, issues

        if dev_info.is_cuda:
            if dev_info.total_vram_gb < 1.8:
                issues.append(
                    PreflightIssue(
                        category="hardware",
                        message=(
                            f"Detected GPU with insufficient VRAM: {dev_info.total_vram_gb:.1f} "
                            "GB. Minimum 2.0 GB required for training."
                        ),
                        critical=True,
                        remediation=(
                            "Use a device with more VRAM or run on CPU with small batch size."
                        ),
                    )
                )
            # Kaggle Dual T4 check on small parameter predictor
            if (
                self.env == "kaggle"
                and dev_info.device_count > 1
                and model_key.lower() == "upn_v2"
                and "RANK" not in os.environ
            ):
                issues.append(
                    PreflightIssue(
                        category="hardware",
                        message=(
                            f"Multi-GPU detected ({dev_info.device_count} GPUs) for lightweight "
                            f"model '{model_key}'. DataParallel causes resource starvation."
                        ),
                        critical=False,
                        remediation="Set CUDA_VISIBLE_DEVICES=0 or preferred_parallel: single.",
                    )
                )

        return dev_info, issues

    def check_dataset_presence(
        self,
        dataset_name: str | None,
        search_roots: list[Path] | None = None,
    ) -> list[PreflightIssue]:
        """Verify that required dataset manifolds exist or are mounted."""
        issues: list[PreflightIssue] = []
        if not dataset_name:
            return issues

        found = False
        roots = search_roots or []
        for r in roots:
            candidate = r / dataset_name
            if candidate.exists():
                found = True
                break

        if not found and roots:
            issues.append(
                PreflightIssue(
                    category="dataset",
                    message=f"Dataset manifold '{dataset_name}' not found in search paths.",
                    critical=True,
                    remediation=(
                        "Verify dataset path in config.yaml or attach dataset in Kaggle/Colab."
                    ),
                )
            )
        return issues

    def run_preflight(
        self,
        working_dir: Path | str,
        model_key: str = "",
        dataset_name: str | None = None,
        search_roots: list[Path] | None = None,
        strict: bool = False,
    ) -> PreflightReport:
        """Run all preflight checks and optionally raise if critical issues found."""
        all_issues: list[PreflightIssue] = []

        all_issues.extend(self.check_packages())
        free_gb, disk_issues = self.check_disk_space(working_dir)
        all_issues.extend(disk_issues)

        dev_info, hw_issues = self.check_hardware(model_key=model_key)
        all_issues.extend(hw_issues)

        if dataset_name and search_roots:
            all_issues.extend(self.check_dataset_presence(dataset_name, search_roots))

        passed = not any(i.critical for i in all_issues)
        report = PreflightReport(
            env=self.env,
            passed=passed,
            issues=tuple(all_issues),
            device_info=dev_info,
            disk_free_gb=free_gb,
        )

        if strict and not passed:
            err_msg = "; ".join(i.message for i in all_issues if i.critical)
            raise RuntimeError(f"Environment preflight validation failed: {err_msg}")

        return report
