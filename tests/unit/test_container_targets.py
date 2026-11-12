"""Container (WebDataset) target shapes and values for every registered model.

Each image model is served from a tiny synthetic tar shard in both shard layouts:

* ``legacy``: labels inside the ``.json`` sidecar (the layout of the published manifolds),
* ``current``: ``.txt`` label sidecars and lossless ``.mask.webp`` masks (the compiler's
  current output).

Models trained outside ``MultiTaskDataset`` (Forex, YOLOv8n, the instruction LLM) are
checked for their routing instead.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from data.container_targets import SUPER_RESOLUTION_MODELS
from data.dataset import MultiTaskDataset
from data.sample_labels import FACE_TARGET_LEN, LANDMARK_ABSENT, MAX_FFANET_BOXES, NIMA_BINS
from tests.fixtures.webdataset import json_bytes, webp_bytes, write_tar
from training import context_data, core_loop

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = yaml.safe_load((PROJECT_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8"))
MODEL_KEYS = sorted(k for k, v in REGISTRY.items() if not k.startswith("_") and isinstance(v, dict))
NON_CONTAINER_MODELS = ("forex_predictor", "yolov8n", "ai_helper_lemgend")
CONTAINER_MODELS = [k for k in MODEL_KEYS if k not in NON_CONTAINER_MODELS]
LAYOUTS = ("legacy", "current")

DISTRIBUTION = [0.0, 0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.05, 0.05, 0.0]
BOX = [0.5, 0.4, 0.2, 0.3]
LANDMARKS = [0.45, 0.35, 0.55, 0.35, 0.5, 0.4, 0.46, 0.46, 0.54, 0.46]
CLASS_INDEX = 2
MASK_CLASS = 7
SAMPLE_KEY = "derain_0001"
SHARD_SIZE = (24, 24)

Members = dict[str, bytes]


def _task(model_key: str) -> str:
    raw = REGISTRY[model_key].get("dataset_type", "restoration")
    return raw[0] if isinstance(raw, list) else raw


def _image_members(paired: bool) -> Members:
    members = {f"{SAMPLE_KEY}.webp": webp_bytes((40, 90, 160), size=SHARD_SIZE, lossless=True)}
    if paired:
        members[f"{SAMPLE_KEY}.target.webp"] = webp_bytes(
            (200, 210, 220), size=SHARD_SIZE, lossless=True
        )
    return members


def _quality_members(layout: str) -> Members:
    if layout == "legacy":
        return {f"{SAMPLE_KEY}.json": json_bytes({"nima_scores": {"distribution": DISTRIBUTION}})}
    return {f"{SAMPLE_KEY}.txt": " ".join(str(v) for v in DISTRIBUTION).encode("utf-8")}


def _face_members(layout: str) -> Members:
    if layout == "legacy":
        points = [LANDMARKS[i : i + 2] for i in range(0, 10, 2)]
        obj = {"bbox_cxcywh_normalized": BOX, "landmarks_normalized": points}
        return {f"{SAMPLE_KEY}.json": json_bytes({"objects": [obj]})}
    line = " ".join(str(v) for v in [0, *BOX, *LANDMARKS])
    return {f"{SAMPLE_KEY}.txt": line.encode("utf-8")}


def _classification_members(layout: str) -> Members:
    if layout == "legacy":
        return {f"{SAMPLE_KEY}.json": json_bytes({"label": CLASS_INDEX})}
    return {f"{SAMPLE_KEY}.txt": str(CLASS_INDEX).encode("utf-8")}


def _segmentation_members(layout: str) -> Members:
    mask = webp_bytes(MASK_CLASS, size=SHARD_SIZE, mode="L", lossless=True)
    meta = (
        {f"{SAMPLE_KEY}.json": json_bytes({"task": "segmentation"})} if layout == "legacy" else {}
    )
    return {f"{SAMPLE_KEY}.mask.webp": mask, **meta}


def _restoration_members(model_key: str, layout: str) -> Members:
    if model_key.startswith("ffanet") and layout == "current":
        return {f"{SAMPLE_KEY}.txt": b"0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1"}
    if layout == "legacy":
        return {f"{SAMPLE_KEY}.json": json_bytes({"task": "restoration"})}
    return {}


def _members(model_key: str, layout: str) -> Members:
    task = _task(model_key)
    unpaired = bool(REGISTRY[model_key].get("needs_synthesis")) and task != "parameter_prediction"
    members = _image_members(paired=not unpaired)
    builders: dict[str, Callable[[str], Members]] = {
        "quality": _quality_members,
        "face_detection": _face_members,
        "classification": _classification_members,
        "segmentation": _segmentation_members,
    }
    builder = builders.get(task)
    members.update(builder(layout) if builder else _restoration_members(model_key, layout))
    return members


def _dataset_from(
    tmp_path: Path, model_key: str, members: Members, is_train: bool = False
) -> MultiTaskDataset:
    """``model_key``'s dataset served from one tar shard holding ``members``."""
    data_root = tmp_path / "datasets"
    shards = data_root / REGISTRY[model_key]["datasets"][0] / "shards"
    shards.mkdir(parents=True)
    write_tar(shards / "shard-000000.tar", members)
    config = {
        "unified_models": "unified_models_v2.yaml",
        "paths": {"datasets_root": str(data_root)},
    }
    return MultiTaskDataset(config, model_key=model_key, is_train=is_train, env="local")


def _dataset(tmp_path: Path, model_key: str, layout: str) -> MultiTaskDataset:
    return _dataset_from(tmp_path, model_key, _members(model_key, layout))


def _tensor_item(dataset: MultiTaskDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Input and tensor target of the dataset's first sample."""
    model_input, target, _ = dataset[0]
    assert isinstance(model_input, torch.Tensor) and isinstance(target, torch.Tensor)
    return model_input, target


def _assert_quality(_model_key: str, layout: str, target: Any, task_id: str) -> None:
    del layout
    assert task_id == "quality"
    assert target.shape == (NIMA_BINS,)
    assert torch.allclose(target, torch.tensor(DISTRIBUTION))


def _assert_face_detection(_model_key: str, layout: str, target: Any, task_id: str) -> None:
    del layout
    assert task_id == "face_detection"
    assert target.shape == (FACE_TARGET_LEN,)
    assert torch.allclose(target, torch.tensor([1.0, *BOX, *LANDMARKS]))


def _assert_classification(_model_key: str, layout: str, target: Any, task_id: str) -> None:
    del layout
    assert task_id == "classification"
    assert target.dtype == torch.long and target.dim() == 0
    assert int(target) == CLASS_INDEX


def _assert_segmentation(_model_key: str, layout: str, target: Any, task_id: str) -> None:
    del layout
    assert task_id == "segmentation"
    assert target.dtype == torch.long
    assert torch.equal(target, torch.full_like(target, MASK_CLASS))


def _assert_parameters(_model_key: str, layout: str, target: Any, task_id: str) -> None:
    del layout
    assert task_id == "parameter_prediction"
    assert target.shape == (3,)
    assert 0.0 <= float(target[0]) <= 1.0 and 0.0 <= float(target[2]) <= 1.0


def _assert_restoration(model_key: str, layout: str, target: Any, task_id: str) -> None:
    if model_key.startswith("ffanet"):
        assert set(target) == {"image", "bboxes"}
        boxes = target["bboxes"]
        assert boxes.shape == (MAX_FFANET_BOXES, 5)
        assert int((boxes.abs().sum(dim=1) > 0).sum()) == (2 if layout == "current" else 0)
        target = target["image"]
    assert target.dim() == 3 and target.shape[0] == 3
    expected = "derain" if model_key == "professional_multitask_restoration" else _task(model_key)
    assert task_id == expected


ASSERTIONS: dict[str, Callable[[str, str, Any, str], None]] = {
    "quality": _assert_quality,
    "face_detection": _assert_face_detection,
    "classification": _assert_classification,
    "segmentation": _assert_segmentation,
    "parameter_prediction": _assert_parameters,
    "restoration": _assert_restoration,
    "face": _assert_restoration,
}


def test_every_registered_model_is_covered() -> None:
    """The registry's 23 models are either container-tested or routed elsewhere."""
    assert len(MODEL_KEYS) == 23
    assert {_task(k) for k in CONTAINER_MODELS} <= set(ASSERTIONS)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("model_key", CONTAINER_MODELS)
def test_container_target(tmp_path: Path, model_key: str, layout: str) -> None:
    """Shapes, dtypes and values of the decoded container triple, per model and layout."""
    dataset = _dataset(tmp_path, model_key, layout)
    assert len(dataset) == 1
    model_input, target, task_id = dataset[0]

    height, width = dataset.size
    scale = SUPER_RESOLUTION_MODELS.get(model_key, 1)
    assert model_input.shape == (3, height // scale, width // scale)
    ASSERTIONS[_task(model_key)](model_key, layout, target, task_id)
    if _task(model_key) in ("restoration", "face"):
        image_target = target["image"] if isinstance(target, dict) else target
        assert image_target.shape == (3, height, width)


UNPAIRED_RESTORATION = [
    k for k in CONTAINER_MODELS if _task(k) == "restoration" and k not in SUPER_RESOLUTION_MODELS
]


@pytest.mark.parametrize("model_key", UNPAIRED_RESTORATION)
def test_unpaired_restoration_never_trains_identity(tmp_path: Path, model_key: str) -> None:
    """A sample without a target member gets a synthesised degradation as its input."""
    dataset = _dataset_from(tmp_path, model_key, {f"{SAMPLE_KEY}.webp": _texture_webp()})
    model_input, target, _ = dataset[0]
    image_target = target["image"] if isinstance(target, dict) else target
    assert not torch.allclose(model_input, image_target)


def _webp_from_pixels(pixels: list[tuple[int, int, int]]) -> bytes:
    image = Image.new("RGB", SHARD_SIZE)
    image.putdata(pixels)
    buf = io.BytesIO()
    image.save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


def _gradient_webp() -> bytes:
    width, height = SHARD_SIZE
    return _webp_from_pixels([(x * 10, y * 10, 128) for y in range(height) for x in range(width)])


def _texture_webp() -> bytes:
    """A high-frequency pattern every synthetic degradation visibly changes."""
    width, height = SHARD_SIZE
    return _webp_from_pixels(
        [
            ((x * 37 + y * 91) % 256, (x * 53 + y * 17) % 256, (x * y * 29) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )


def test_ultrazoom_input_is_aligned_downscale(tmp_path: Path) -> None:
    """The super-resolution input is the aligned target crop downscaled by the factor."""
    gradient = _gradient_webp()
    members = {f"{SAMPLE_KEY}.webp": gradient, f"{SAMPLE_KEY}.target.webp": gradient}
    model_input, target = _tensor_item(_dataset_from(tmp_path, "ultrazoom", members, True))
    expected = F.interpolate(
        target.unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True
    )
    assert torch.allclose(model_input, expected.squeeze(0).clamp(0.0, 1.0), atol=1e-5)


def test_face_box_only_label_masks_landmarks(tmp_path: Path) -> None:
    """A YOLO face label without landmarks is a positive face with absent landmarks."""
    members = _image_members(paired=False)
    members[f"{SAMPLE_KEY}.txt"] = " ".join(str(v) for v in [0, *BOX]).encode("utf-8")
    _, target = _tensor_item(_dataset_from(tmp_path, "retinaface", members))
    assert float(target[0]) == 1.0
    assert torch.allclose(target[1:5], torch.tensor(BOX))
    assert torch.all(target[5:] == LANDMARK_ABSENT)


def test_segmentation_without_mask_raises(tmp_path: Path) -> None:
    """A segmentation sample without a mask member fails loudly instead of all-zero masks."""
    dataset = _dataset_from(tmp_path, "parsenet", _image_members(paired=False))
    with pytest.raises(RuntimeError, match="no mask member"):
        dataset.getitem_container(dataset.catalog.samples[0][0], 0, seed=0)


def _spec(model_key: str) -> context_data.RunSpec:
    return context_data.RunSpec(
        args=argparse.Namespace(env="local"),
        config={},
        model_info=REGISTRY[model_key],
        model_key=model_key,
        project_root=PROJECT_ROOT,
        task_type=_task(model_key),
        device=torch.device("cpu"),
    )


def test_forex_and_llm_route_to_their_own_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forex and the instruction LLM never reach the container ``MultiTaskDataset``."""
    calls: list[str] = []
    monkeypatch.setattr(context_data, "_forex_datasets", lambda *_: calls.append("forex"))
    monkeypatch.setattr(context_data, "_instruction_datasets", lambda *_: calls.append("llm"))
    monkeypatch.setattr(context_data, "_image_datasets", lambda *_: calls.append("image"))
    context_data.build_datasets(_spec("forex_predictor"), None, 1.0)
    context_data.build_datasets(_spec("ai_helper_lemgend"), None, 1.0)
    assert calls == ["forex", "llm"]


def test_yolov8n_routes_to_the_ultralytics_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """YOLOv8n is intercepted before a ``TrainingContext`` (and its datasets) is built."""
    calls: list[str] = []
    monkeypatch.setattr(core_loop, "load_secrets", lambda: None)
    monkeypatch.setattr(core_loop, "load_workspace_config", lambda _root: {})
    monkeypatch.setattr(
        core_loop, "run_governed_yolo_training", lambda *_: calls.append("ultralytics")
    )

    def _no_context(_args: Any) -> None:
        raise AssertionError("yolov8n must not build a TrainingContext")

    monkeypatch.setattr(core_loop, "build_training_context", _no_context)
    core_loop.main(["--model", "yolov8n"])
    assert calls == ["ultralytics"]


def test_registry_has_no_duplicate_datasets() -> None:
    """Every model lists each dataset once (audit 3.1.7)."""
    for model_key in MODEL_KEYS:
        datasets = REGISTRY[model_key].get("datasets", [])
        assert len(datasets) == len(set(datasets)), model_key
    forex = REGISTRY["_registry_metadata"]["datasets"]["LemGendizedForexUniverse"]
    assert forex["count"] == 26818700
