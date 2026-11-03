"""In-Process Training Service for LemGendary Model Training Suite.

Orchestrates training lifecycle, parameter resolution, hardware policies,
and execution loops without subprocess self-invocation.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from training.core_loop import _run_yolo_native, build_training_context
from training.services.notebook_service import NotebookService
from training.training.engine import TrainingSummary, run_training
from training.utils.paths import get_project_root

logger = logging.getLogger("lemtrain.services.training_service")


@dataclass(frozen=True)
class TrainHyperparams:
    """Run-size overrides; ``None`` keeps the preset/config value.

    Attributes:
        epochs: Total training epochs (overrides preset/config if specified).
        batch_size: Physical batch size per gradient step.
        lr: Base learning rate.
        preset: Optional preset profile name from presets.yaml.
        resolution: Selected input spatial resolution or timeframe horizon.
    """

    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    preset: str | None = None
    resolution: int | None = None


@dataclass(frozen=True)
class TrainRequest:
    """One in-process training run.

    Attributes:
        model_key: Target model key registered in unified_models.yaml.
        hyper: Epochs, batch size, learning rate, preset and resolution overrides.
        env: Execution environment ('local', 'kaggle', 'colab').
        clean: Whether to ignore prior checkpoints and start fresh from epoch 1.
        auto_sync: Enable automated cloud synchronization.
        parallel: Parallel strategy ('auto', 'single', 'dp', 'ddp').
        enable_sawtooth: Enable Sawtooth VRAM Governor memory protection.
    """

    model_key: str
    hyper: TrainHyperparams = TrainHyperparams()
    env: str = "local"
    clean: bool = False
    auto_sync: bool = False
    parallel: str = "auto"
    enable_sawtooth: bool = True

    @classmethod
    def from_options(cls, model_key: str, **options: Any) -> TrainRequest:
        """Build a request from flat keyword options (hyperparameters and run flags mixed)."""
        hyper_names = {f.name for f in fields(TrainHyperparams)}
        hyper = TrainHyperparams(**{k: v for k, v in options.items() if k in hyper_names})
        flags = {k: v for k, v in options.items() if k not in hyper_names}
        return cls(model_key=model_key, hyper=hyper, **flags)


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

        presets: dict[str, Any] = {}
        if self.presets_path.exists():
            with open(self.presets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    loaded = data.get("presets")
                    if isinstance(loaded, dict):
                        presets = loaded

        self._cached_presets = presets
        return presets

    def get_preset(self, preset_name: str) -> dict[str, Any] | None:
        """Retrieve a specific preset configuration by name."""
        presets = self.load_presets()
        return presets.get(preset_name)

    def _sync_notebooks(self, model_key: str) -> None:
        """Regenerate the model's notebooks; failures only skip this pre-flight step."""
        try:
            nb_service = NotebookService(project_root=self.project_root)
            nb_service.generate_notebooks(model_key=model_key, platform="all")
        except (OSError, yaml.YAMLError, ValueError, KeyError, TypeError, AttributeError) as e:
            logger.warning("Pre-flight notebook generation skipped for %s: %s", model_key, e)

    def _with_preset(self, hyper: Any) -> tuple[Any, Any, Any]:
        """``(epochs, batch_size, lr)``; unset values come from ``hyper.preset`` if given."""
        epochs, batch_size, lr = hyper.epochs, hyper.batch_size, hyper.lr
        if hyper.preset:
            preset_cfg = self.get_preset(hyper.preset)
            if preset_cfg:
                if epochs is None:
                    epochs = preset_cfg.get("epochs")
                if batch_size is None:
                    batch_size = preset_cfg.get("batch_size")
                if lr is None:
                    lr = preset_cfg.get("learning_rate")
        return epochs, batch_size, lr

    def train(
        self,
        request: TrainRequest,
        *,
        on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> TrainingSummary:
        """Execute a training job in-process.

        Args:
            request: Model key and run options (see :class:`TrainRequest`).
            on_epoch_end: Optional callback invoked after each epoch.
            cancel_check: Optional callable returning True when cancellation is requested.

        Returns:
            TrainingSummary: Completed run metrics and summary.
        """
        model_key, hyper = request.model_key, request.hyper
        # 0. Regenerate and sync fresh notebooks for this model across LemGendaryModels, manifolds,
        # and training dirs
        self._sync_notebooks(model_key)

        # 1. Resolve preset parameters if requested
        epochs, batch_size, lr = self._with_preset(hyper)

        # 2. Build synthetic CLI namespace for deterministic context builder
        args = argparse.Namespace(
            model=model_key,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            env=request.env,
            prefetch_datasets="",
            hub_user=None,
            hub_repo=None,
            auto_sync=request.auto_sync,
            reset_scheduler=False,
            clean=request.clean,
            phase=1,
            fold=1,
            pairs=None,
            timeframes=(
                [hyper.resolution] if hyper.resolution and model_key == "forex_predictor" else None
            ),
            resolution=hyper.resolution,
            enable_sawtooth=request.enable_sawtooth,
            num_workers=None,
            val_num_workers=None,
            enable_batch_growth=False,
            parallel=request.parallel,
        )

        # 3. Intercept yolov8n before building TrainingContext - it uses Ultralytics native trainer
        if model_key == "yolov8n":
            config_path = self.project_root / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return _run_yolo_native(
                args,
                config,
                self.project_root,
                on_epoch_end=on_epoch_end,
                cancel_check=cancel_check,
            )

        # 4. Construct deterministic TrainingContext
        ctx = build_training_context(args)

        # 5. Run training coordinator
        summary = run_training(ctx, on_epoch_end=on_epoch_end, cancel_check=cancel_check)
        return summary
