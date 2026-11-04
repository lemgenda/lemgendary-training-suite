"""SOTA Exporter: Checkpoint to Standalone PyTorch (Legacy CLI Shim)."""

from __future__ import annotations

import logging

from export.export_common import LegacyExportCli, run_legacy_export_cli

logger = logging.getLogger("lemtrain.export.torch_cli")

_CLI = LegacyExportCli(
    description="LemGendary SOTA Exporter: Checkpoint to Standalone PyTorch",
    banner="Initializing Standalone PyTorch Exporter for model: %s",
    failure="PyTorch standalone export failed: %s",
    targets=("torch_pt",),
    log=logger,
)


def main() -> None:
    """CLI entrypoint for standalone PyTorch model export."""
    run_legacy_export_cli(_CLI)


if __name__ == "__main__":
    main()
