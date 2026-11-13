"""Resume semantics of the run settings (R4-C5).

A paused run finishes the current epoch on the recorded settings and the next epoch
picks up the configured ones; frozen settings refuse a resume unless ``--clean`` or
``--override-frozen`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset

from training.context_resume import pin_recorded_settings
from training.core_loop import build_cli_parser
from training.data.loaders import (
    LoaderPlacement,
    TrainLoaderOptions,
    WorkerOverrides,
    build_train_loader,
)
from training.data.locality import LocalityConfig, shard_window_order
from training.data.locality_resume import LEGACY_LOCALITY
from training.data.locality_resume import PAYLOAD_KEY as OLD_LOCALITY_KEY
from training.data.resumable_sampler import (
    ResumableBatchSampler,
    resume_start_batch,
    set_loader_epoch,
)
from training.governance.curriculum import CurriculumState
from training.run_settings.capture import (
    SettingsInputs,
    epoch_settings,
    frozen_settings,
    param_group_layout,
)
from training.run_settings.keys import BATCH, LOCALITY, LR_SCHEDULE
from training.run_settings.ledger import SETTINGS_LEDGER_NAME, SettingsLedger
from training.run_settings.pinning import pin_config_settings
from training.run_settings.policy import (
    FrozenPolicy,
    FrozenSettingsChanged,
    RunSettingsTracker,
)
from training.run_settings.record import (
    PAYLOAD_KEY,
    RunSettingsError,
    RunSettingsRecord,
    SettingChange,
    settings_sha,
)
from training.services.training_service import ResumeOptions, TrainRequest
from training.training.context import TrainingContext
from training.training.lr_scaling import APPLIED_MULT_KEY
from training.training.optimizer import build_scheduler
from training.training.settings_switch import apply_batch, apply_due_settings, rebuild_scheduler

SHARDS = 8
PER_SHARD = 10
KEYS = [f"shard-{i // PER_SHARD}" for i in range(SHARDS * PER_SHARD)]
BATCH_SIZE = 5


class _ShardedDataset(Dataset[int]):
    """Index dataset that reports a shard per sample."""

    def __len__(self) -> int:
        return len(KEYS)

    def __getitem__(self, index: int) -> int:
        return index

    def sample_shard_keys(self) -> list[str]:
        """Shard of each index."""
        return list(KEYS)


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "batch_size": None,
        "lr": None,
        "num_workers": None,
        "val_num_workers": None,
        "fold": 1,
        "pairs": None,
        "timeframes": None,
        "clean": False,
        "override_frozen": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _config(window: int = 4, block: int = 256, **sections: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "data": {"locality": {"window_shards": window, "fraction_block_size": block}}
    }
    config.update(sections)
    return config


def _inputs(
    config: dict[str, Any],
    model_info: dict[str, Any] | None = None,
    task_type: str = "restoration",
    args: argparse.Namespace | None = None,
) -> SettingsInputs:
    return SettingsInputs(
        config=config,
        model_info=model_info or {"kwargs": {"width": 8}, "datasets": ["Set"]},
        args=args or _args(),
        task_type=task_type,
        total_epochs=10,
        model_class="Net",
        param_groups=(("backbone", 2), ("backbone", 1)),
    )


def _tracker(inputs: SettingsInputs, ledger_dir: Path | None = None) -> RunSettingsTracker:
    ledger = SettingsLedger.in_folder(ledger_dir) if ledger_dir is not None else None
    return RunSettingsTracker(frozen_settings(inputs), epoch_settings(inputs), ledger=ledger)


def _payload(inputs: SettingsInputs, epoch: int) -> dict[str, Any]:
    return {"epoch": epoch, PAYLOAD_KEY: _tracker(inputs).record(epoch).to_payload()}


def _ledger_lines(folder: Path) -> list[dict[str, Any]]:
    path = folder / SETTINGS_LEDGER_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _loader(config: dict[str, Any]) -> Any:
    return build_train_loader(
        _ShardedDataset(),
        BATCH_SIZE,
        options=TrainLoaderOptions(drop_last=False, workers=WorkerOverrides(num_workers=0)),
        placement=LoaderPlacement(config=config),
    )


def _order(loader: Any) -> list[int]:
    return [int(i) for batch in loader for i in batch]


def _seed(loader: Any) -> int:
    sampler = loader.batch_sampler
    assert isinstance(sampler, ResumableBatchSampler)
    return sampler.seed


def _context(
    tracker: RunSettingsTracker,
    config: dict[str, Any],
    **parts: Any,
) -> TrainingContext:
    return cast(
        TrainingContext,
        SimpleNamespace(
            state=SimpleNamespace(run_settings=tracker),
            run=SimpleNamespace(config=config, args=parts.get("args", _args())),
            models=SimpleNamespace(
                optimizer=parts.get("optimizer"), scheduler=parts.get("scheduler")
            ),
            governance=SimpleNamespace(
                governor=SimpleNamespace(curriculum=parts.get("curriculum", CurriculumState()))
            ),
            data=SimpleNamespace(accumulation_steps=1, train_loader=parts.get("loader")),
        ),
    )


# --- frozen settings -------------------------------------------------------------------


def test_frozen_hash_is_stable_and_order_independent() -> None:
    """The same settings hash the same, whatever their key order."""
    frozen = frozen_settings(_inputs(_config()))
    reordered = dict(reversed(list(frozen.items())))
    assert settings_sha(frozen) == settings_sha(reordered)
    assert settings_sha(frozen) == settings_sha(frozen_settings(_inputs(_config())))
    record = RunSettingsRecord(frozen, {}, 3)
    assert RunSettingsRecord.from_payload({PAYLOAD_KEY: record.to_payload()}) == record


def test_tampered_record_is_rejected() -> None:
    """A frozen group that does not match its hash raises instead of being trusted."""
    payload = RunSettingsRecord({"seed": 1}, {}, 1).to_payload()
    payload["frozen"]["seed"] = 2
    with pytest.raises(RunSettingsError):
        RunSettingsRecord.from_payload({PAYLOAD_KEY: payload})


def test_frozen_change_is_refused_naming_the_diff_and_clean() -> None:
    """A changed architecture refuses the resume; the message names it and --clean."""
    old = _inputs(_config(), {"kwargs": {"width": 8}})
    new = _inputs(_config(), {"kwargs": {"width": 16}})
    with pytest.raises(FrozenSettingsChanged) as refused:
        _tracker(new).resume(_payload(old, 3), 3, FrozenPolicy(), mid_epoch=False)
    assert "model.kwargs" in str(refused.value)
    assert "--clean" in str(refused.value) and "--override-frozen" in str(refused.value)


def test_frozen_change_allowed_with_override_and_recorded(tmp_path: Path) -> None:
    """--override-frozen resumes and writes an override line to the ledger."""
    old = _inputs(_config(), {"kwargs": {"width": 8}})
    new = _inputs(_config(), {"kwargs": {"width": 16}})
    tracker = _tracker(new, tmp_path)
    tracker.resume(_payload(old, 3), 3, FrozenPolicy(override=True), mid_epoch=False)
    assert tracker.frozen["model.kwargs"] == {"width": 16}
    (line,) = _ledger_lines(tmp_path)
    assert line["kind"] == "override_frozen"
    assert line["changes"]["model.kwargs"] == {"from": {"width": 8}, "to": {"width": 16}}


def test_frozen_change_allowed_with_clean() -> None:
    """A fresh run (--clean) does not check the old record."""
    old = _inputs(_config(), {"kwargs": {"width": 8}})
    new = _inputs(_config(), {"kwargs": {"width": 16}})
    policy = FrozenPolicy.from_args(_args(clean=True))
    _tracker(new).resume(_payload(old, 3), 3, policy, mid_epoch=False)


def test_override_flag_reaches_the_policy() -> None:
    """``--override-frozen`` parses into the frozen policy."""
    args = build_cli_parser().parse_args(["--override-frozen"])
    assert FrozenPolicy.from_args(args) == FrozenPolicy(clean=False, override=True)


def test_emd_form_and_forex_risk_are_frozen() -> None:
    """NIMA ``stabilizers.emd_form`` and forex ``evaluation.risk_per_trade`` are frozen."""
    nima = {"stabilizers": {"emd_form": "nima"}}
    squared = {"stabilizers": {"emd_form": "squared"}}
    with pytest.raises(FrozenSettingsChanged, match="stabilizers.emd_form"):
        _tracker(_inputs(_config(), squared, "quality")).resume(
            _payload(_inputs(_config(), nima, "quality"), 2), 2, FrozenPolicy(), mid_epoch=True
        )
    one, two = {"evaluation": {"risk_per_trade": 0.01}}, {"evaluation": {"risk_per_trade": 0.02}}
    with pytest.raises(FrozenSettingsChanged, match="evaluation.risk_per_trade"):
        _tracker(_inputs(_config(), two, "forex")).resume(
            _payload(_inputs(_config(), one, "forex"), 2), 2, FrozenPolicy(), mid_epoch=False
        )


def test_param_group_layout_counts_parameters_per_role() -> None:
    """The layout is ``(role, parameter count)`` per group."""
    groups = [{"params": [1, 2], "group": "backbone"}, {"params": [3], "group": "head"}]
    assert param_group_layout(groups) == (("backbone", 2), ("head", 1))


# --- epoch settings across a resume ----------------------------------------------------


def test_mid_epoch_resume_replays_the_recorded_order_then_switches(tmp_path: Path) -> None:
    """The rest of the epoch replays the recorded order; the next epoch uses the new one."""
    recorded = _inputs(_config(window=2))
    config = _config(window=4)
    args = _args()
    tracker = _tracker(_inputs(config, args=args), tmp_path)

    pinned = tracker.resume(_payload(recorded, 3), 3, FrozenPolicy(), mid_epoch=True)
    pin_config_settings(config, args, pinned)
    assert LocalityConfig.from_config(config) == LocalityConfig(2, 256)

    loader = _loader(config)
    set_loader_epoch(loader, 3)
    recorded_order = shard_window_order(KEYS, 2, seed=_seed(loader) + 3)
    start = resume_start_batch(loader, 4)
    assert _order(loader) == recorded_order[start * BATCH_SIZE :]

    def rebuild(ctx: TrainingContext) -> None:
        ctx.data.train_loader = _loader(ctx.run.config)

    ctx = _context(tracker, config, args=args, loader=loader)
    apply_due_settings(ctx, 3, rebuild)
    assert ctx.data.train_loader is loader, "nothing switches inside the resumed epoch"
    apply_due_settings(ctx, 4, rebuild)
    new_loader = ctx.data.train_loader
    set_loader_epoch(new_loader, 4)
    assert _order(new_loader) == shard_window_order(KEYS, 4, seed=_seed(new_loader) + 4)
    assert tracker.active[LOCALITY] == {"window_shards": 4, "fraction_block_size": 256}
    assert tracker.record(4).epoch[LOCALITY]["window_shards"] == 4

    (line,) = _ledger_lines(tmp_path)
    assert line["epoch"] == 4 and line["kind"] == "epoch_switch"
    assert line["changes"][LOCALITY]["from"]["window_shards"] == 2
    assert line["changes"][LOCALITY]["to"]["window_shards"] == 4


def test_epoch_boundary_resume_applies_at_the_first_epoch(tmp_path: Path) -> None:
    """After a completed epoch the configured settings apply as the next epoch starts."""
    config = _config(window=4)
    tracker = _tracker(_inputs(config), tmp_path)
    tracker.resume(_payload(_inputs(_config(window=2)), 5), 5, FrozenPolicy(), mid_epoch=False)
    assert tracker.pending is not None and tracker.pending.apply_from == 6
    rebuilt: list[bool] = []
    apply_due_settings(_context(tracker, config), 6, lambda ctx: rebuilt.append(True))
    assert rebuilt == [True]
    assert LocalityConfig.from_config(config) == LocalityConfig(4, 256)
    assert _ledger_lines(tmp_path)[0]["epoch"] == 6


def test_unchanged_settings_resume_without_a_switch(tmp_path: Path) -> None:
    """Matching settings leave nothing pending and write no ledger line."""
    inputs = _inputs(_config())
    tracker = _tracker(inputs, tmp_path)
    tracker.resume(_payload(inputs, 2), 2, FrozenPolicy(), mid_epoch=True)
    assert tracker.pending is None
    assert not (tmp_path / SETTINGS_LEDGER_NAME).exists()


def test_old_checkpoint_without_the_record(tmp_path: Path) -> None:
    """No frozen check; the epoch finishes on the old locality, then the config applies."""
    config = _config(window=4)
    new = _inputs(_config(), {"kwargs": {"anything": "changed"}})
    tracker = _tracker(new, tmp_path)
    old_record = {OLD_LOCALITY_KEY: {"window_shards": 8, "fraction_block_size": 32}}
    pinned = tracker.resume({"epoch": 2, **old_record}, 2, FrozenPolicy(), mid_epoch=True)
    assert tracker.legacy_checkpoint
    assert pinned[LOCALITY] == {"window_shards": 8, "fraction_block_size": 32}
    apply_due_settings(_context(tracker, config), 3, lambda ctx: None)
    assert _ledger_lines(tmp_path)[0]["kind"] == "legacy_checkpoint"

    oldest = _tracker(new)
    pinned = oldest.resume({"epoch": 2}, 2, FrozenPolicy(), mid_epoch=True)
    assert pinned[LOCALITY] == LEGACY_LOCALITY.as_dict()


# --- batch size ------------------------------------------------------------------------


def _batch_change(batch: int | None, target: int = 24) -> SettingChange:
    old = {"batch_size": 6, "target_effective_batch": 24, "accumulation_steps": None}
    new = {"batch_size": batch, "target_effective_batch": target, "accumulation_steps": None}
    return SettingChange(old, new)


def test_batch_is_accepted_only_as_an_exact_divisor() -> None:
    """Batch 8 of 24 gives accumulation 3; batch 7 is refused and the old batch kept."""
    curriculum = CurriculumState(current_batch=6, current_acc=4, target_effective_batch=24)
    ctx = _context(RunSettingsTracker({}, {}), _config(), curriculum=curriculum)
    assert apply_batch(ctx, _batch_change(8))
    assert (curriculum.current_batch, curriculum.current_acc) == (8, 3)
    assert ctx.data.accumulation_steps == 3
    assert not apply_batch(ctx, _batch_change(7))
    assert (curriculum.current_batch, curriculum.current_acc) == (8, 3)


def test_refused_batch_stays_recorded_and_is_logged(tmp_path: Path) -> None:
    """A refused batch keeps the recorded value active and lands in ``refused``."""
    recorded = _inputs(_config(), {"batch_size": 6})
    current = _inputs(_config(), {"batch_size": 7})
    tracker = _tracker(current, tmp_path)
    tracker.resume(_payload(recorded, 1), 1, FrozenPolicy(), mid_epoch=False)
    curriculum = CurriculumState(current_batch=6, current_acc=4, target_effective_batch=24)
    apply_due_settings(_context(tracker, _config(), curriculum=curriculum), 2, lambda ctx: None)
    assert tracker.active[BATCH]["batch_size"] == 6
    (line,) = _ledger_lines(tmp_path)
    assert BATCH in line["refused"] and BATCH not in line["changes"]


# --- LR schedule -----------------------------------------------------------------------


def _optimizer(lr: float) -> torch.optim.Optimizer:
    layer = torch.nn.Linear(2, 1)
    return torch.optim.SGD(
        [{"params": [layer.weight]}, {"params": [layer.bias], APPLIED_MULT_KEY: 0.5}], lr=lr
    )


def _run_epochs(optimizer: torch.optim.Optimizer, scheduler: Any, epochs: int) -> None:
    for _ in range(epochs):
        optimizer.step()
        scheduler.step()


def _schedule(kind: str, lr: float, total: int) -> dict[str, Any]:
    return {"scheduler": {"type": kind, "warmup_epochs": 2}, "lr": lr, "total_epochs": total}


@pytest.mark.parametrize("kind", ["cosine", "warmup_cosine"])
def test_scheduler_is_rebuilt_and_fast_forwarded(kind: str) -> None:
    """The new schedule runs from the steps already taken, as if it had been used from 0."""
    optimizer = _optimizer(0.1)
    old = build_scheduler(optimizer, {"scheduler": {"type": "cosine"}}, total_epochs=10)
    _run_epochs(optimizer, old, 3)

    new = rebuild_scheduler(optimizer, 3, _schedule(kind, 0.01, 20))

    reference_opt = torch.optim.SGD(
        [
            {"params": [torch.zeros(1, requires_grad=True)], "lr": 0.01},
            {"params": [torch.zeros(1, requires_grad=True)], "lr": 0.005},
        ],
        lr=0.01,
    )
    reference = build_scheduler(
        reference_opt, {"scheduler": _schedule(kind, 0.01, 20)["scheduler"]}, total_epochs=20
    )
    _run_epochs(reference_opt, reference, 3)
    assert new.last_epoch == 3
    for group, expected in zip(optimizer.param_groups, reference_opt.param_groups, strict=True):
        assert group["lr"] == pytest.approx(expected["lr"])
    assert isinstance(new, CosineAnnealingLR if kind == "cosine" else LambdaLR)
    if isinstance(new, CosineAnnealingLR):
        assert new.T_max == 20
    _run_epochs(optimizer, new, 1)
    _run_epochs(reference_opt, reference, 1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(reference_opt.param_groups[0]["lr"])


def test_mid_epoch_resume_loads_scheduler_state_into_the_recorded_schedule() -> None:
    """A changed schedule is rebuilt from the record before its state is loaded."""
    recorded = _inputs(_config(), args=_args(lr=0.1))
    config = _config(scheduler={"type": "step", "step_size": 3})
    args = _args(lr=0.1)
    tracker = _tracker(_inputs(config, args=args))
    optimizer = _optimizer(0.1)
    train = SimpleNamespace(
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, config, total_epochs=10),
    )
    spec = SimpleNamespace(args=args, config=config)
    pin_recorded_settings(cast(Any, spec), cast(Any, train), _payload(recorded, 4), tracker)
    assert isinstance(train.scheduler, CosineAnnealingLR)
    due = tracker.due(5)
    assert due is not None and LR_SCHEDULE in due.changes


def test_train_request_groups_the_resume_flags() -> None:
    """``clean`` and ``override_frozen`` options land in the request's resume options."""
    request = TrainRequest.from_options("m", epochs=2, clean=True, override_frozen=True)
    assert request.resume == ResumeOptions(clean=True, override_frozen=True)
    assert request.hyper.epochs == 2
