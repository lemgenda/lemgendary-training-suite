"""In-Process Training Service for LemGendary Model Training Suite.

Orchestrates training lifecycle, parameter resolution, hardware policies,
and execution loops without subprocess self-invocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable
import yaml

from training.core_loop import build_training_context
from training.training.context import TrainingContext
from training.training.engine import TrainingSummary, run_training
from training.utils.paths import get_project_root


class TrainingService:
    """Orchestrates model training workflows in-process."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or get_project_root()
        self.presets_path = self.project_root / "presets.yaml"
        self._cached_presets: dict[str, Any] | None = None

    def load_presets(self) -> dict[str, Any]:
        """Load canonical presets from presets.yaml."""
        if self._cached_presets is not None:
            return self._cached_presets

        if self.presets_path.exists():
            with open(self.presets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._cached_presets = data.get("presets", {})
        else:
            self._cached_presets = {}

        return self._cached_presets

    def get_preset(self, preset_name: str) -> dict[str, Any] | None:
        """Retrieve a specific preset configuration by name."""
        presets = self.load_presets()
        return presets.get(preset_name)

    def train(
        self,
        model_key: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        preset: str | None = None,
        env: str = "local",
        clean: bool = False,
        auto_sync: bool = False,
        parallel: str = "auto",
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
    ) -> TrainingSummary:
        """Execute a training job in-process.

        Args:
            model_key: Target model key registered in unified_models.yaml.
            epochs: Total training epochs (overrides preset/config if specified).
            batch_size: Physical batch size per gradient step.
            lr: Base learning rate.
            preset: Optional preset profile name from presets.yaml.
            env: Execution environment ('local', 'kaggle', 'colab').
            clean: Whether to ignore prior checkpoints and start fresh from epoch 1.
            auto_sync: Enable automated cloud synchronization.
            parallel: Parallel strategy ('auto', 'single', 'dp', 'ddp').
            on_epoch_end: Optional callback invoked after each epoch.

        Returns:
            TrainingSummary: Completed run metrics and summary.
        """
        # 1. Resolve preset parameters if requested
        if preset:
            preset_cfg = self.get_preset(preset)
            if preset_cfg:
                if epochs is None:
                    epochs = preset_cfg.get("epochs")
                if batch_size is None:
                    batch_size = preset_cfg.get("batch_size")
                if lr is None:
                    lr = preset_cfg.get("learning_rate")

        # 2. Build synthetic CLI namespace for deterministic context builder
        args = argparse.Namespace(
            model=model_key,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            env=env,
            prefetch_datasets="",
            hub_user=None,
            hub_repo=None,
            auto_sync=auto_sync,
            reset_scheduler=False,
            clean=clean,
            phase=1,
            fold=1,
            pairs=None,
            timeframes=None,
            num_workers=None,
            val_num_workers=None,
            enable_batch_growth=False,
            parallel=parallel,
        )

        # 3. Construct deterministic TrainingContext
        ctx = build_training_context(args)

        # 4. Run training coordinator
        summary = run_training(ctx)
        return summary
