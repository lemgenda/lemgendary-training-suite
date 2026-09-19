"""Evaluation Service for LemGendary Model Training Suite.

Executes deterministic validation and metric evaluation passes over model checkpoints
in-process under torch.no_grad().
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import torch
import yaml

from training.checkpoint.manager import safe_load_checkpoint
from training.checkpoint.recovery import CheckpointRecoveryEngine
from training.core_loop import build_training_context
from training.training.validation import validate_one_epoch
from training.utils.paths import get_project_root


class EvaluationService:
    """Evaluates trained model checkpoints and computes validation metrics."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()

    def evaluate(
        self,
        model_key: str,
        checkpoint_path: Path | str | None = None,
        batch_size: int | None = None,
        env: str = "local",
    ) -> dict[str, Any]:
        """Run in-process evaluation on a model checkpoint.

        Args:
            model_key: Target model key registered in unified models registry.
            checkpoint_path: Optional explicit path to checkpoint file (.pth/.pt).
            batch_size: Evaluation batch size.
            env: Execution environment ('local', 'kaggle', 'colab').

        Returns:
            dict[str, Any]: Evaluation summary and metric dictionary.
        """
        # 1. Build synthetic CLI namespace to construct context
        args = argparse.Namespace(
            model=model_key,
            epochs=1,
            batch_size=batch_size,
            lr=None,
            env=env,
            prefetch_datasets="",
            hub_user=None,
            hub_repo=None,
            auto_sync=False,
            reset_scheduler=False,
            clean=False,
            phase=1,
            fold=1,
            pairs=None,
            timeframes=None,
            num_workers=0,
            val_num_workers=0,
            enable_batch_growth=False,
            parallel="single",
        )

        ctx = build_training_context(args)

        # 2. If explicit checkpoint provided, load weights
        resolved_ckpt: Path | None = None
        if checkpoint_path is not None:
            resolved_ckpt = Path(checkpoint_path).resolve()
            if resolved_ckpt.exists():
                ckpt_data = safe_load_checkpoint(resolved_ckpt, map_location=ctx.device_info.device)
                if ckpt_data and "model_state" in ckpt_data:
                    ctx.model.load_state_dict(ckpt_data["model_state"], strict=False)
                elif ckpt_data and "state_dict" in ckpt_data:
                    ctx.model.load_state_dict(ckpt_data["state_dict"], strict=False)
        else:
            recovery_engine = CheckpointRecoveryEngine(model_key=model_key)
            resolved_ckpt = recovery_engine.find_best_checkpoint()
            if resolved_ckpt and resolved_ckpt.exists():
                ckpt_data = safe_load_checkpoint(resolved_ckpt, map_location=ctx.device_info.device)
                if ckpt_data and "model_state" in ckpt_data:
                    ctx.model.load_state_dict(ckpt_data["model_state"], strict=False)

        # 3. Execute validation pass
        metrics = validate_one_epoch(ctx, epoch=1)

        result: dict[str, Any] = {
            "model_key": model_key,
            "checkpoint_path": str(resolved_ckpt) if resolved_ckpt else None,
            "metrics": metrics,
            "device": str(ctx.device_info.device),
        }
        return result
