"""SOTA Exporter: Checkpoint to FP32/FP16 ONNX (Legacy CLI Shim)."""

from __future__ import annotations

import logging

from export.export_common import LegacyExportCli, run_legacy_export_cli

logger = logging.getLogger("lemtrain.export.onnx_cli")

_CLI = LegacyExportCli(
    description="LemGendary SOTA Exporter: Checkpoint to FP32/FP16 ONNX",
    banner="Initializing SOTA ONNX Exporter for model: %s",
    failure="ONNX export failed: %s",
    targets=("onnx_fp32", "onnx_fp16"),
    log=logger,
)


def main() -> None:
    """CLI entrypoint for ONNX model export."""
    run_legacy_export_cli(_CLI)


if __name__ == "__main__":
    main()
