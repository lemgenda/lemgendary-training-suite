"""Unlabeled classification samples are ignored, not trained as class 0 (R4-FU item 3).

A sample whose label names no class gets ``IGNORE_TARGET``: the cross-entropy skips it, the
validation accuracy leaves it out and ``catalog.unlabeled`` counts it per dataset (logged at
1, 2, 4, ...). Label decoding itself is unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from torch import nn

from data.dataset import MultiTaskDataset
from data.dataset_targets import DirectorySample, SamplePaths
from tests.fixtures.webdataset import json_bytes, single_shard_config, webp_bytes
from training.context_data import RunSpec
from training.core_loop import build_criterion
from training.ignored_targets import (
    IGNORE_TARGET,
    IgnoreTargetCollision,
    UnlabeledLog,
    check_class_count,
    check_class_id,
)
from training.losses import CombinedLoss
from training.training.val_accumulators import ValAccumulators
from training.training.val_metrics import ValBatch, update_classification

MODEL_KEY = "universal_nsfw_classification"
_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
_IMAGE = webp_bytes((40, 90, 160), size=(24, 24), lossless=True)


def _dataset(tmp_path: Path, members: dict[str, bytes]) -> MultiTaskDataset:
    name = _REGISTRY[MODEL_KEY]["datasets"][0]
    config = single_shard_config(tmp_path / "datasets", name, members)
    return MultiTaskDataset(config, model_key=MODEL_KEY, is_train=True, env="local")


def _target(item: tuple) -> int:
    """The class target of a dataset item."""
    target = item[1]
    assert isinstance(target, torch.Tensor) and target.dtype == torch.long
    return int(target)


def test_container_sample_without_label_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No label in the shard: the target is ``IGNORE_TARGET`` and the sample is counted."""
    dataset = _dataset(tmp_path, {"a_0001.webp": _IMAGE, "b_0002.webp": _IMAGE})
    with caplog.at_level(logging.WARNING, logger="lemtrain.data.unlabeled"):
        targets = [_target(dataset[i]) for i in range(2)]
    assert targets == [IGNORE_TARGET, IGNORE_TARGET]
    name = _REGISTRY[MODEL_KEY]["datasets"][0]
    assert dataset.catalog.unlabeled.count(name) == 2
    assert "2 unlabeled classification samples so far" in caplog.text


def test_container_labelled_sample_keeps_its_class(tmp_path: Path) -> None:
    """A labelled sample still decodes to its registry id."""
    dataset = _dataset(tmp_path, {"a_0001.webp": _IMAGE, "a_0001.json": json_bytes({"label": 3})})
    assert _target(dataset[0]) == 3
    assert dataset.catalog.unlabeled.count(_REGISTRY[MODEL_KEY]["datasets"][0]) == 0


@pytest.mark.parametrize(("fname", "expected"), [("x.png", IGNORE_TARGET), ("2/x.png", 2)])
def test_directory_sample_without_label_or_class_folder(
    tmp_path: Path, fname: str, expected: int
) -> None:
    """No label file and no class folder: ignored; a numeric class folder still names a class."""
    dataset = _dataset(tmp_path, {"a_0001.webp": _IMAGE})
    ds_path = tmp_path / "flat"
    paths = SamplePaths(fname=fname, ds_path=str(ds_path), img_path="", tgt_path="")
    sample = DirectorySample(paths=paths, task_str="classification", img_tensor=torch.zeros(3))
    assert _target(dataset.classification_item(sample)) == expected


def test_loss_skips_unlabeled_samples() -> None:
    """The loss equals the cross-entropy of the labelled samples; none labelled gives 0."""
    criterion = CombinedLoss("classification")
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, IGNORE_TARGET, 4])
    expected = nn.functional.cross_entropy(logits[[0, 2]], torch.tensor([1, 4]))
    torch.testing.assert_close(criterion(logits, target), expected)
    empty = criterion(logits, torch.full((3,), IGNORE_TARGET))
    assert float(empty.detach()) == 0.0
    empty.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_accuracy_leaves_unlabeled_samples_out() -> None:
    """Validation accuracy counts only the labelled samples."""
    acc = ValAccumulators()
    preds = torch.tensor([[0.0, 9.0], [9.0, 0.0], [9.0, 0.0]])
    batch = ValBatch(
        preds=preds,
        targets=torch.tensor([1, IGNORE_TARGET, 1]),
        sota_cfg={"accuracy": 0.9},
        dataset_type="classification",
        device=torch.device("cpu"),
        governor=None,
    )
    update_classification(acc, batch)
    assert (acc.tasks.total_cls_correct, acc.tasks.total_cls_samples) == (1, 2)


def test_log_reports_powers_of_two(caplog: pytest.LogCaptureFixture) -> None:
    """The running count is logged at 1, 2, 4, 8 occurrences."""
    log = UnlabeledLog()
    with caplog.at_level(logging.WARNING, logger="lemtrain.data.unlabeled"):
        for i in range(9):
            log.record("ds", "m", f"s{i}")
    counts = [r.args[2] for r in caplog.records if isinstance(r.args, tuple)]
    assert counts == [1, 2, 4, 8]


def _classifier(count: int) -> dict[str, Any]:
    return {"kwargs": {"num_classes": count}}


def test_class_count_must_leave_the_ignore_target_free() -> None:
    """Up to IGNORE_TARGET classes (ids below it) pass; one more is refused clearly."""
    check_class_count(_classifier(IGNORE_TARGET), "wide")
    check_class_count({"class_names": [f"c{i}" for i in range(IGNORE_TARGET)]}, "named")
    check_class_count({}, "undeclared")
    with pytest.raises(IgnoreTargetCollision, match="declares 256 classes"):
        check_class_count(_classifier(IGNORE_TARGET + 1), "too_wide")
    with pytest.raises(IgnoreTargetCollision, match="declares 300 classes"):
        check_class_count({"class_names": [f"c{i}" for i in range(300)]}, "too_wide")


def test_criterion_setup_refuses_too_many_classes() -> None:
    """Building a classifier's criterion fails before training starts."""
    spec = SimpleNamespace(
        model_info=_classifier(IGNORE_TARGET + 1),
        model_key="too_wide",
        task_type="classification",
        args=SimpleNamespace(model="too_wide"),
        device=torch.device("cpu"),
    )
    with pytest.raises(IgnoreTargetCollision):
        build_criterion(cast(RunSpec, spec))


def test_class_id_at_the_ignore_target_is_refused() -> None:
    """A resolved class id equal to IGNORE_TARGET would be skipped, so it raises."""
    assert check_class_id(IGNORE_TARGET - 1, "s") == IGNORE_TARGET - 1
    with pytest.raises(IgnoreTargetCollision, match="Class id 255 of s"):
        check_class_id(IGNORE_TARGET, "s")


def test_directory_class_folder_at_the_ignore_target_raises(tmp_path: Path) -> None:
    """Without registry class names, a class folder ``255`` raises instead of being ignored."""
    dataset = _dataset(tmp_path, {"a_0001.webp": _IMAGE})
    dataset.spec = dataclasses.replace(dataset.spec, class_names=())
    paths = SamplePaths(fname="255/x.png", ds_path=str(tmp_path), img_path="", tgt_path="")
    sample = DirectorySample(paths=paths, task_str="classification", img_tensor=torch.zeros(3))
    with pytest.raises(IgnoreTargetCollision):
        dataset.classification_item(sample)
