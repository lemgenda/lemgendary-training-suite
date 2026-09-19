"""Export Service for LemGendary Model Training Suite.

Orchestrates multi-target model artifact compilation (ONNX FP32/FP16, Standalone PyTorch,
WebGPU, and MT5 Forex) in-process without shell spawning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.export import export_all
from training.utils.paths import get_project_root


class ExportService:
    """Orchestrates production model export pipelines."""

    SUPPORTED_TARGETS: list[str] = [
        "onnx_fp32",
        "onnx_fp16",
        "torch_pt",
        "webgpu",
        "forex",
    ]

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()

    def list_supported_targets(self) -> list[str]:
        """Return list of supported export target identifiers."""
        return list(self.SUPPORTED_TARGETS)

    def export(
        self,
        model_key: str,
        checkpoint_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        targets: list[str] | None = None,
    ) -> dict[str, str]:
        """Compile and export model binaries.

        Args:
            model_key: Model key from unified models registry.
            checkpoint_path: Optional explicit path to checkpoint file.
            output_dir: Optional destination directory for exported binaries.
            targets: Specific target subset (e.g. ['onnx_fp32', 'torch_pt']).

        Returns:
            dict[str, str]: Mapping of target name to exported file path.
        """
        target_list = targets or ["onnx_fp32", "onnx_fp16", "torch_pt"]
        for t in target_list:
            if t not in self.SUPPORTED_TARGETS:
                raise ValueError(
                    f"Unsupported export target '{t}'. Supported: {self.SUPPORTED_TARGETS}"
                )

        exported_paths = export_all(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            targets=target_list,
        )

        return {target: str(path) for target, path in exported_paths.items()}
