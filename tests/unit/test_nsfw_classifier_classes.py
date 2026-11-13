"""Universal NSFW classifier: 5 registry classes end to end.

The registry entry declares ``class_names`` (ids 0 drawings, 1 hentai, 2 neutral, 3 porn,
4 sexy) and ``kwargs.num_classes: 5``. The factory builds a 5-logit head from it, one reader
(:mod:`training.registry.class_names`) validates the names, the label decoders accept ids
0..4 and the names, and the model card and the ``.pt`` export metadata list the classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml
from torch import nn
from torchvision import models as tv_models

from data.dataset import MultiTaskDataset
from data.sample_labels import ClassLabelRangeError, class_index
from models.factory import get_model
from tests.fixtures.webdataset import json_bytes, single_shard_config, webp_bytes
from training.core_loop import load_model_info
from training.export.pytorch_exporter import pt_metadata
from training.model_card import build_model_readme
from training.model_card_snippets import SnippetInput, usage_snippet
from training.registry import get_model_registry
from training.registry.class_names import ClassNamesError, class_names
from training.registry.preprocessing import UNIT

MODEL_KEY = "universal_nsfw_classification"
NSFW_CLASSES = ("drawings", "hentai", "neutral", "porn", "sexy")
_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
_real_efficientnet = tv_models.efficientnet_v2_s


def _efficientnet_without_download(*_args: Any, **kwargs: Any) -> nn.Module:
    """EfficientNetV2-S with random weights: the pretrained download is skipped."""
    kwargs["weights"] = None
    return _real_efficientnet(**kwargs)


def _build_classifier() -> nn.Module:
    """The registry classifier through the real factory, without network weight downloads."""
    with mock.patch.object(tv_models, "efficientnet_v2_s", _efficientnet_without_download):
        model = get_model(MODEL_KEY)
    return model.eval()


def test_factory_builds_five_class_head() -> None:
    """``get_model`` sizes the head from ``kwargs.num_classes``: 5 logits per image."""
    model = _build_classifier()
    head = model.get_submodule("backbone.classifier.1")
    assert isinstance(head, nn.Linear) and head.out_features == len(NSFW_CLASSES)
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2, len(NSFW_CLASSES))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, len(NSFW_CLASSES) - 1]))
    assert torch.isfinite(loss)


def test_registry_class_names_are_read_and_validated() -> None:
    """The reader returns the names in id order and agrees with ``num_classes``."""
    info = _REGISTRY[MODEL_KEY]
    assert class_names(info) == NSFW_CLASSES
    assert info["kwargs"]["num_classes"] == len(NSFW_CLASSES)
    assert class_names(load_model_info(_ROOT, {}, MODEL_KEY)) == NSFW_CLASSES
    assert not class_names({"kwargs": {"num_classes": 3}})
    assert not class_names(None)


@pytest.mark.parametrize(
    "info",
    [
        {"class_names": ["a", "b"], "kwargs": {"num_classes": 3}},
        {"class_names": ["a", "a"]},
        {"class_names": ["a", " "]},
        {"class_names": []},
        {"class_names": "a b"},
        {"class_names": ["a", True]},
        {"class_names": ["a"], "kwargs": {"num_classes": 0}},
    ],
)
def test_malformed_class_names_are_rejected(info: dict[str, Any]) -> None:
    """A mismatched count, duplicate, blank or non-string name is a ``ClassNamesError``."""
    with pytest.raises(ClassNamesError):
        class_names(info)


def test_every_registry_classifier_is_consistent() -> None:
    """Every entry declaring ``class_names`` passes the reader (and ``num_classes`` matches)."""
    for key, info in _REGISTRY.items():
        if isinstance(info, dict) and "class_names" in info:
            assert class_names(info), key


@pytest.mark.parametrize("idx", range(len(NSFW_CLASSES)))
def test_label_ids_zero_to_four_decode(idx: int) -> None:
    """Ids 0..4 decode as int, whole float, numeric text and class name."""
    for label in (idx, float(idx), str(idx), f" {idx}.0\n", NSFW_CLASSES[idx]):
        assert class_index(label, NSFW_CLASSES) == idx


@pytest.mark.parametrize("label", [5, "5", -1, "-1"])
def test_label_ids_outside_the_classes_are_rejected(label: Any) -> None:
    """An id outside 0..4 is a typed error, never a silent class."""
    with pytest.raises(ClassLabelRangeError):
        class_index(label, NSFW_CLASSES)


@pytest.mark.parametrize("label", [None, True, 2.5, "nan", "unknown", b"2"])
def test_labels_naming_no_class_decode_to_none(label: Any) -> None:
    """Missing, boolean, fractional and unknown labels name no class."""
    assert class_index(label, NSFW_CLASSES) is None


def _dataset(tmp_path: Path, members: dict[str, bytes]) -> MultiTaskDataset:
    """The classifier's dataset served from one WebDataset shard holding ``members``."""
    dataset_name = _REGISTRY[MODEL_KEY]["datasets"][0]
    config = single_shard_config(tmp_path / "datasets", dataset_name, members)
    return MultiTaskDataset(config, model_key=MODEL_KEY, is_train=False, env="local")


def _members(label: dict[str, bytes]) -> dict[str, bytes]:
    return {"nsfw_0001.webp": webp_bytes((40, 90, 160), size=(24, 24), lossless=True), **label}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ({"nsfw_0001.json": json_bytes({"label": 4})}, 4),
        ({"nsfw_0001.json": json_bytes({"label": "porn"})}, 3),
        ({"nsfw_0001.txt": b"0"}, 0),
        ({"nsfw_0001.txt": b"hentai"}, 1),
    ],
)
def test_container_labels_decode_to_registry_ids(
    tmp_path: Path, label: dict[str, bytes], expected: int
) -> None:
    """Container samples decode numeric and named labels to the registry ids."""
    dataset = _dataset(tmp_path, _members(label))
    assert dataset.spec.class_names == NSFW_CLASSES
    _, target, task_id = dataset[0]
    assert task_id == "classification"
    assert isinstance(target, torch.Tensor) and target.dtype == torch.long
    assert int(target) == expected


def test_container_label_outside_the_classes_raises(tmp_path: Path) -> None:
    """A sample labelled 5 fails at decoding instead of in the loss."""
    dataset = _dataset(tmp_path, _members({"nsfw_0001.json": json_bytes({"label": 5})}))
    with pytest.raises(ClassLabelRangeError):
        _ = dataset[0]


def test_model_card_lists_the_classes() -> None:
    """The card shows the classes, top-1 only (top-5 is trivial for 5 classes) and usage."""
    card = build_model_readme(MODEL_KEY, _REGISTRY, 3, {"acc": 0.91})
    legend = "".join(f"> - `{idx}`: {name}\n" for idx, name in enumerate(NSFW_CLASSES))
    assert legend in card
    assert "**Top-1 Accuracy**: 0.91" in card
    assert "Top-5" not in card
    assert f"CLASS_NAMES = {list(NSFW_CLASSES)!r}" in card
    assert all(len(line) <= 80 for line in card.splitlines() if line.startswith(">"))


def test_usage_snippet_without_names_prints_ids() -> None:
    """A classifier without registry names gets an id-only snippet."""
    snippet = usage_snippet("classification", "toy", SnippetInput(32, 32))
    assert "CLASS_NAMES = None" in snippet
    assert 'get_model("toy")' in snippet


def test_pt_export_metadata_carries_the_class_names() -> None:
    """The ``.pt`` metadata maps logit indices to class names; other models carry none."""
    registry = get_model_registry()
    metadata = pt_metadata(MODEL_KEY, registry.get_model(MODEL_KEY))
    assert metadata == {
        "model_key": MODEL_KEY,
        "preprocessing": UNIT.metadata(),
        "class_names": list(NSFW_CLASSES),
    }
    assert pt_metadata("ultrazoom", registry.get_model("ultrazoom")) == {
        "model_key": "ultrazoom",
        "preprocessing": UNIT.metadata(),
    }
