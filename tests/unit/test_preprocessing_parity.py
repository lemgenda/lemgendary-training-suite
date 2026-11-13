"""One image preprocessing per model: training, validation, export, card and cells agree.

``training.registry.preprocessing.image_preprocessing`` is the single reader; the dataset
transforms, the ``.pt`` export metadata, the model card and the generated inference cells
all follow it (R4-FU item 2: the NSFW cells applied ImageNet statistics to a model trained
on ``[0, 1]`` pixels).
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from data.dataset import MultiTaskDataset
from tests.fixtures.webdataset import json_bytes, single_shard_config, webp_bytes
from training.core_loop import load_model_info
from training.model_card import build_model_readme
from training.notebooks.cells.inference import (
    build_onnx_fp16_cell,
    build_onnx_fp32_cell,
    build_pth_cell,
)
from training.notebooks.registry import resolve_model_metadata
from training.registry.preprocessing import (
    IMAGENET,
    UNIT,
    ImagePreprocessing,
    PreprocessingError,
    image_preprocessing,
)
from training.registry.preprocessing_source import (
    indented,
    numpy_input_source,
    torch_input_source,
)

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: dict[str, Any] = yaml.safe_load(
    (_ROOT / "unified_models_v2.yaml").read_text(encoding="utf-8")
)
NSFW = "universal_nsfw_classification"
NIMA = "nima_aesthetic_mobile"
CASES = [(NSFW, UNIT), (NIMA, IMAGENET), ("ultrazoom", UNIT)]


@pytest.mark.parametrize(("model_key", "expected"), CASES)
def test_registry_preprocessing(model_key: str, expected: ImagePreprocessing) -> None:
    """Classifiers and restoration train on [0, 1]; quality (NIMA) is ImageNet-normalized."""
    assert image_preprocessing(_REGISTRY[model_key]) == expected
    assert image_preprocessing(load_model_info(_ROOT, {}, model_key)) == expected


def test_registry_key_overrides_the_task_default() -> None:
    """An explicit ``preprocessing.normalization`` wins over the task type."""
    info = {"dataset_type": "classification", "preprocessing": {"normalization": "imagenet"}}
    assert image_preprocessing(info) == IMAGENET
    assert image_preprocessing({"dataset_type": "quality"}) == IMAGENET
    assert image_preprocessing(None) == UNIT


@pytest.mark.parametrize(
    "raw", [{"normalization": "zscore"}, {"normalization": None}, "imagenet", ["unit"]]
)
def test_invalid_preprocessing_is_rejected(raw: object) -> None:
    """An unknown normalization or a non-mapping value is a typed error at load time."""
    with pytest.raises(PreprocessingError):
        image_preprocessing({"preprocessing": raw})


def _image(size: int) -> Image.Image:
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def _dataset(tmp_path: Path, model_key: str) -> MultiTaskDataset:
    """The model's validation dataset served from a one-sample WebDataset shard."""
    members = {
        "s_0001.webp": webp_bytes((40, 90, 160), size=(24, 24), lossless=True),
        "s_0001.json": json_bytes({"label": 0}),
    }
    name = _REGISTRY[model_key]["datasets"][0]
    config = single_shard_config(tmp_path / "datasets", name, members)
    return MultiTaskDataset(config, model_key=model_key, is_train=False, env="local")


@pytest.mark.parametrize(("model_key", "expected"), CASES[:2])
def test_cells_reproduce_the_validation_transform(
    tmp_path: Path, model_key: str, expected: ImagePreprocessing
) -> None:
    """The generated NumPy and PyTorch input code equals the dataset's validation transform."""
    dataset = _dataset(tmp_path, model_key)
    assert dataset.spec.preprocessing == expected
    img = _image(dataset.size[0])
    want = dataset.pipes.transform(img).unsqueeze(0).numpy()

    script = tmp_path / "cell_input.py"
    script.write_text(
        "".join([*numpy_input_source(expected), *torch_input_source(expected)]),
        encoding="utf-8",
    )
    scope = runpy.run_path(
        str(script), init_globals={"np": np, "torch": torch, "img": img, "device": "cpu"}
    )
    np.testing.assert_allclose(scope["input_data"], want, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scope["input_tensor"].numpy(), want, rtol=1e-6, atol=1e-6)
    assert scope["input_data"].dtype == np.float32


@pytest.mark.parametrize(("model_key", "expected"), CASES)
def test_notebook_cells_use_the_model_preprocessing(
    model_key: str, expected: ImagePreprocessing
) -> None:
    """Every image inference cell renders the model's preprocessing, nothing else."""
    meta = resolve_model_metadata(model_key, _REGISTRY)
    assert meta.preprocessing == expected
    numpy_lines = indented(numpy_input_source(expected))
    for build in (build_onnx_fp32_cell, build_onnx_fp16_cell):
        source = "".join(build(meta)["source"])
        assert "".join(numpy_lines) in source
        assert ("0.485" in source) is expected.normalizes
    pth_source = "".join(build_pth_cell(meta)["source"])
    assert "".join(indented(torch_input_source(expected))) in pth_source
    assert ("0.485" in pth_source) is expected.normalizes


@pytest.mark.parametrize(("model_key", "expected"), CASES)
def test_model_card_states_the_preprocessing(model_key: str, expected: ImagePreprocessing) -> None:
    """The card's input requirements and usage snippet follow the model's preprocessing."""
    card = build_model_readme(model_key, _REGISTRY, 3, {})
    assert ("ImageNet stats" in card) is False
    assert ("0.485" in card) is expected.normalizes
