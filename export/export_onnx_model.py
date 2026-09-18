"""SOTA Exporter: Checkpoint to FP32/FP16 ONNX (Legacy CLI Shim)."""

from __future__ import annotations

import argparse
import logging
import sys

from training.export import ExportError, export_all
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.export.onnx_cli")


def main() -> None:
    """CLI entrypoint for ONNX model export."""
    parser = argparse.ArgumentParser(description="LemGendary SOTA Exporter: Checkpoint to FP32/FP16 ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model key from unified_models.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific .pth checkpoint to export")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts")
    args = parser.parse_args()

    logger.info("Initializing SOTA ONNX Exporter for model: %s", args.model)
    try:
        results = export_all(
            model_key=args.model,
            checkpoint_path=args.checkpoint,
            targets=["onnx_fp32", "onnx_fp16"],
        )
        for target, path in results.items():
            logger.info("Exported %s -> %s", target, path)
    except ExportError as err:
        logger.error("ONNX export failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
