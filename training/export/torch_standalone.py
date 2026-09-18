"""Standalone PyTorch model export for zero-dependency inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.export.common import ExportError

logger = logging.getLogger("lemtrain.export.torch_standalone")


def export_torch_standalone(
    model: nn.Module,
    save_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Export model object and state dict to standalone .pt file."""
    out_file = Path(save_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    raw_model = getattr(model_to_save, "model", model_to_save)
    raw_model.eval()

    save_payload: dict[str, Any] = {
        "model_state": raw_model.state_dict(),
        "model": raw_model,
    }
    if metadata:
        save_payload["metadata"] = dict(metadata)

    logger.info("Saving standalone PyTorch model to %s...", out_file)
    try:
        torch.save(save_payload, str(out_file))
        logger.info("Successfully exported standalone model to %s", out_file)
        return True
    except Exception as exc:
        err_msg = f"Standalone PyTorch export to {out_file} failed: {exc}"
        logger.error(err_msg)
        raise ExportError("TORCH_STANDALONE_EXPORT_FAILED", err_msg) from exc
