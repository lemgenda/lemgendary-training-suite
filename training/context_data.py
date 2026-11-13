"""Datasets, loaders and batch size for ``build_training_context``.

``RunSpec`` carries the resolved run inputs (CLI arguments, configs, task type and device);
the builders here turn it into the train and validation datasets and their loaders.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from data.dataset import MultiTaskDataset
from data.forex_dataset import ForexDataset, ForexSplit
from data.forex_sampler import RowGroupAwareSampler
from training.data import build_train_loader, rebuild_train_loader
from training.data import build_val_loader as canonical_build_val_loader
from training.data.instruction_dataset import (
    InstructionDataset,
    collate_instruction_batch,
)
from training.data.loaders import (
    LoaderPlacement,
    TrainLoaderOptions,
    ValLoaderOptions,
    WorkerOverrides,
)
from training.data.resumable_sampler import ResumableBatchSampler
from training.hardware.probe import VramProbe, audit_hardware_vram

logger = logging.getLogger("lemtrain.core_loop")

LLM_TASKS = ("text_instruction", "llm")

DEFAULT_DATASETS_ROOT = "../LemGendaryDatasets"
"""``paths.datasets_root`` when ``config.yaml`` does not set it (relative to the project root)."""

DEFAULT_INSTRUCTION_DATASET = "LemGendizedAIHelperLemGend"
"""Instruction dataset used when the model entry declares no ``datasets``."""

# Failure modes of dataset construction (missing shards, malformed manifests, bad config values).
DATASET_BUILD_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
)


@dataclass(frozen=True)
class RunSpec:
    """Resolved inputs of one training run."""

    args: argparse.Namespace
    config: dict
    model_info: dict
    model_key: str
    project_root: Path
    task_type: str
    device: Any

    @property
    def is_llm(self) -> bool:
        """True for instruction-tuned language models (always trained on all data)."""
        return self.task_type in LLM_TASKS or self.model_info.get("category") == "llm"


def build_instruction_train_loader(
    dataset: InstructionDataset, batch_size: int
) -> DataLoader[dict[str, Any]]:
    """Build the instruction-tuning train loader with dynamic-padding collation."""
    return DataLoader(
        dataset=dataset,
        batch_sampler=ResumableBatchSampler(len(dataset), batch_size, shuffle=True),
        collate_fn=collate_instruction_batch,
    )


def _cli_batch_size(raw_batch: Any) -> int | None:
    """Parse ``--batch_size``; ``auto``, ``none``, blanks and non-positive values mean unset."""
    if raw_batch is None:
        return None
    if isinstance(raw_batch, str) and raw_batch.strip().lower() in ("auto", "none", ""):
        return None
    try:
        parsed_bs = int(raw_batch)
    except (ValueError, TypeError):
        return None
    return parsed_bs if parsed_bs > 0 else None


def requested_batch_size(args: Any, model_info: dict[str, Any]) -> int | None:
    """CLI batch size, else the model's configured one; ``None`` for ``auto`` (VRAM audit)."""
    resolved_batch = _cli_batch_size(getattr(args, "batch_size", None))
    if resolved_batch is not None:
        return resolved_batch
    cfg_batch = model_info.get("batch_size")
    if isinstance(cfg_batch, bool):
        return None
    if isinstance(cfg_batch, int) and cfg_batch > 0:
        return cfg_batch
    if isinstance(cfg_batch, str) and cfg_batch.strip().isdigit() and int(cfg_batch) > 0:
        return int(cfg_batch)
    return None


def resolve_batch_size(spec: RunSpec, raw_model: Any) -> int:
    """CLI batch size, else the model's configured one, else the VRAM audit's safe batch."""
    resolved_batch = requested_batch_size(spec.args, spec.model_info)
    if resolved_batch is None:
        resolved_batch = audit_hardware_vram(
            model_key=spec.model_key,
            model_info=spec.model_info,
            config=spec.config,
            model=raw_model,
            probe=VramProbe(
                device=spec.device,
                res_override=getattr(spec.args, "resolution", None),
                mode="train",
                fold=getattr(spec.args, "fold", None),
                pairs=getattr(spec.args, "pairs", None) or spec.model_info.get("pairs"),
            ),
        )

    if not isinstance(resolved_batch, int) or resolved_batch <= 0:
        resolved_batch = 128 if spec.task_type == "forex" else 16
    return resolved_batch


def _forex_datasets(spec: RunSpec, sample_fraction: float):
    """Walk-forward Forex train and validation shards (validation is optional)."""
    shard_root = str(spec.config.get("paths", {}).get("datasets_root", DEFAULT_DATASETS_ROOT))
    kwargs = spec.model_info.get("kwargs", {})
    pairs = getattr(spec.args, "pairs", None) or spec.model_info.get("pairs") or kwargs.get("pairs")
    timeframes = (
        getattr(spec.args, "timeframes", None)
        or spec.model_info.get("active_timeframes")
        or kwargs.get("active_timeframes")
    )
    fold = getattr(spec.args, "fold", 1)
    train_ds = ForexDataset(
        shard_root,
        ForexSplit(
            pairs=pairs,
            active_timeframes=timeframes,
            is_train=True,
            sample_fraction=sample_fraction,
            fold=fold,
        ),
    )
    try:
        val_ds = ForexDataset(
            shard_root,
            ForexSplit(pairs=pairs, active_timeframes=timeframes, is_train=False, fold=fold),
        )
    except DATASET_BUILD_ERRORS as forex_val_err:
        logger.warning(
            "Forex val dataset could not be built; running without validation: %s",
            forex_val_err,
        )
        val_ds = None
    return train_ds, val_ds


def datasets_root(spec: RunSpec) -> Path:
    """``paths.datasets_root`` of ``config.yaml``; relative values start at the project root."""
    root = Path(str(spec.config.get("paths", {}).get("datasets_root") or DEFAULT_DATASETS_ROOT))
    return root if root.is_absolute() else spec.project_root / root


def instruction_jsonl_paths(spec: RunSpec) -> tuple[Path, Path]:
    """Locate the train and validation JSONL files of an instruction dataset."""
    train_jsonl, val_jsonl, _ = instruction_all_jsonl_paths(spec)
    return train_jsonl, val_jsonl


def instruction_all_jsonl_paths(spec: RunSpec) -> tuple[Path, Path, Path | None]:
    """Locate the train, validation, and optional test JSONL files of an instruction dataset."""
    declared_datasets = spec.model_info.get("datasets", [])
    dataset_name = declared_datasets[0] if declared_datasets else DEFAULT_INSTRUCTION_DATASET
    data_dir = datasets_root(spec) / dataset_name / "data"

    train_jsonl = data_dir / f"{spec.model_key}_train.jsonl"
    val_jsonl = data_dir / f"{spec.model_key}_val.jsonl"
    test_jsonl = data_dir / f"{spec.model_key}_test.jsonl"
    if not train_jsonl.exists():
        # Fall back to generic *_train.jsonl / *_val.jsonl candidates in data/.
        candidates_train = list(data_dir.glob("*_train.jsonl")) if data_dir.exists() else []
        candidates_val = list(data_dir.glob("*_val.jsonl")) if data_dir.exists() else []
        candidates_test = list(data_dir.glob("*_test.jsonl")) if data_dir.exists() else []
        if candidates_train:
            train_jsonl = candidates_train[0]
        if candidates_val:
            val_jsonl = candidates_val[0]
        if candidates_test:
            test_jsonl = candidates_test[0]
    if not train_jsonl.exists():
        # Fall back to the default ai_helper files.
        train_jsonl = data_dir / "ai_helper_train.jsonl"
        val_jsonl = data_dir / "ai_helper_val.jsonl"
        test_jsonl = data_dir / "ai_helper_test.jsonl"
    return train_jsonl, val_jsonl, (test_jsonl if test_jsonl.exists() else None)


def _instruction_datasets(spec: RunSpec, raw_model: Any, sample_fraction: float):
    """Instruction-tuning JSONL datasets tokenized with the model's tokenizer."""
    train_jsonl, val_jsonl, test_jsonl = instruction_all_jsonl_paths(spec)
    tokenizer = getattr(raw_model, "tokenizer", None)
    max_length = int(spec.model_info.get("input_size", 2048))
    train_ds = InstructionDataset(
        train_jsonl,
        tokenizer=tokenizer,
        max_length=max_length,
        sample_fraction=sample_fraction,
    )
    val_ds = (
        InstructionDataset(val_jsonl, tokenizer=tokenizer, max_length=max_length)
        if val_jsonl.exists()
        else None
    )
    test_ds = (
        InstructionDataset(test_jsonl, tokenizer=tokenizer, max_length=max_length)
        if test_jsonl and test_jsonl.exists()
        else None
    )
    return train_ds, val_ds, test_ds


def _image_datasets(spec: RunSpec, sample_fraction: float):
    """Multi-task image datasets; an empty or broken validation split is skipped."""
    train_ds = MultiTaskDataset(
        spec.config,
        model_key=spec.model_key,
        is_train=True,
        env=spec.args.env,
        sample_fraction=sample_fraction,
    )
    try:
        val_ds = MultiTaskDataset(
            spec.config,
            model_key=spec.model_key,
            is_train=False,
            env=spec.args.env,
            sample_fraction=1.0,
        )
        if len(val_ds) == 0:
            logger.info("Validation dataset contains 0 samples; running without validation.")
            val_ds = None
    except DATASET_BUILD_ERRORS as val_err:
        logger.warning(
            "Validation dataset could not be built; running without validation: %s", val_err
        )
        val_ds = None
    return train_ds, val_ds


def build_datasets(spec: RunSpec, raw_model: Any, sample_fraction: float):
    """Return ``(train_ds, val_ds)`` or ``(train_ds, val_ds, test_ds)`` for task type."""
    if spec.task_type == "forex":
        return _forex_datasets(spec, sample_fraction)
    if spec.task_type in LLM_TASKS:
        return _instruction_datasets(spec, raw_model, sample_fraction)
    return _image_datasets(spec, sample_fraction)


def _sampler_kwargs(spec: RunSpec, train_ds: Any) -> dict[str, Any]:
    """Forex runs read row groups in locality order through ``RowGroupAwareSampler``."""
    if spec.task_type != "forex":
        return {}
    return {"sampler": RowGroupAwareSampler(train_ds, shuffle=True, seed=42), "shuffle": False}


def _placement(spec: RunSpec) -> LoaderPlacement:
    """Device, environment and config the run's loaders are tuned for."""
    return LoaderPlacement(device=spec.device, env=spec.args.env, config=spec.config)


def _worker_overrides(spec: RunSpec) -> WorkerOverrides:
    """``--num_workers`` override for the train loader (``None`` keeps the topology default)."""
    return WorkerOverrides(num_workers=spec.args.num_workers)


def build_initial_train_loader(
    spec: RunSpec, train_ds: Any, batch_size: int, sample_fraction: float
):
    """Build the first train loader for ``train_ds``."""
    if isinstance(train_ds, InstructionDataset):
        return build_instruction_train_loader(train_ds, batch_size)
    sampler_kwargs = _sampler_kwargs(spec, train_ds)
    if sampler_kwargs:
        logger.info(
            "RowGroupAwareSampler active for Forex dataset: %d locality groups (fraction: %.2f)",
            sampler_kwargs["sampler"].num_groups,
            sample_fraction,
        )
    return build_train_loader(
        dataset=train_ds,
        batch_size=batch_size,
        options=TrainLoaderOptions(**sampler_kwargs, workers=_worker_overrides(spec)),
        placement=_placement(spec),
    )


def rebuild_resumed_train_loader(spec: RunSpec, train_ds: Any, old_loader: Any, batch_size: int):
    """Rebuild the train loader after a resume changed the dataset or batch size."""
    return rebuild_train_loader(
        old_loader=old_loader,
        dataset=train_ds,
        batch_size=batch_size,
        options=TrainLoaderOptions(
            **_sampler_kwargs(spec, train_ds), workers=_worker_overrides(spec)
        ),
        placement=_placement(spec),
    )


def build_val_loader_for(spec: RunSpec, val_ds: Any, batch_size: int):
    """Build the validation loader, or ``None`` without (or on failure to load) a val split."""
    if val_ds is None or len(val_ds) <= 0:
        return None
    try:
        return canonical_build_val_loader(
            dataset=val_ds,
            batch_size=batch_size,
            options=ValLoaderOptions(
                workers=WorkerOverrides(
                    # 0 or unset means "auto": the worker topology sizes the loader.
                    num_workers=spec.args.val_num_workers or spec.args.num_workers or None
                ),
                collate_fn=collate_instruction_batch if spec.task_type in LLM_TASKS else None,
            ),
            placement=_placement(spec),
        )
    except DATASET_BUILD_ERRORS as val_loader_err:
        logger.warning(
            "Validation loader could not be built; running without validation: %s",
            val_loader_err,
        )
        return None
