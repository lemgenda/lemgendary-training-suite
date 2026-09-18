"""SOTA Exporter: Checkpoint to Standalone PyTorch (Legacy CLI Shim)."""

from __future__ import annotations

import argparse
import logging
import sys

from training.export import ExportError, export_all

logger = logging.getLogger("lemtrain.export.torch_cli")


def main() -> None:
    """CLI entrypoint for standalone PyTorch model export."""
    parser = argparse.ArgumentParser(description="LemGendary SOTA Exporter: Checkpoint to Standalone PyTorch")
    parser.add_argument("--model", type=str, required=True, help="Model key from unified_models.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific .pth checkpoint to export")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts")
    args = parser.parse_args()

    logger.info("Initializing Standalone PyTorch Exporter for model: %s", args.model)
    try:
        results = export_all(
            model_key=args.model,
            checkpoint_path=args.checkpoint,
            targets=["torch_pt"],
        )
        for target, path in results.items():
            logger.info("Exported %s -> %s", target, path)
    except ExportError as err:
        logger.error("PyTorch standalone export failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
