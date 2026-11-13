"""Exported image-model ONNX files carry their metadata (R4-FU item 4).

``metadata_props`` hold the model key, task, ``[C, H, W]`` input size, the training
preprocessing and, for a classifier, the class names in logit order. Tiny stand-in models
keep the export fast; the FP32 file keeps its external-data sidecar loadable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from training.export.onnx_metadata import onnx_metadata, read_metadata, stamp_metadata
from training.export.pytorch_exporter import PyTorchModelExporter
from training.registry import get_model_registry
from training.registry.preprocessing import IMAGENET, UNIT, ImagePreprocessing
from training.utils.lazy_imports import load_onnx, load_onnxruntime

NSFW = "universal_nsfw_classification"
NSFW_CLASSES = ["drawings", "hentai", "neutral", "porn", "sexy"]
_SIZE = 32


def _classifier() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)
    ).eval()


def _export(model: nn.Module, model_key: str, out: Path) -> dict[str, Path]:
    sample = torch.rand(1, 3, _SIZE, _SIZE)
    return PyTorchModelExporter().export(model, model_key, output_dir=out, dummy_input=sample)


def test_classifier_onnx_files_carry_class_names(tmp_path: Path) -> None:
    """Both ONNX release variants describe the NSFW classifier; the FP32 file still runs."""
    files = _export(_classifier(), NSFW, tmp_path)
    for key in ("onnx_fp32", "onnx_fp16"):
        props = read_metadata(files[key])
        assert json.loads(props["class_names"]) == NSFW_CLASSES
        assert props["model_key"] == NSFW
        assert props["task"] == "classification"
        assert json.loads(props["input_size"]) == [3, _SIZE, _SIZE]
        assert props["normalization"] == UNIT.name
        assert props["pixel_scale"] == "1/255"
    session = load_onnxruntime().InferenceSession(
        str(files["onnx_fp32"]), providers=["CPUExecutionProvider"]
    )
    feed = {session.get_inputs()[0].name: np.zeros((1, 3, _SIZE, _SIZE), np.float32)}
    assert session.run(None, feed)[0].shape == (1, len(NSFW_CLASSES))


@pytest.mark.parametrize(
    ("model_key", "normalization"), [("ultrazoom", UNIT), ("nima_technical", IMAGENET)]
)
def test_other_image_models_carry_preprocessing_only(
    model_key: str, normalization: ImagePreprocessing
) -> None:
    """Models without registry class names get no ``class_names`` entry."""
    spec = get_model_registry().get_model(model_key)
    assert spec is not None
    props = onnx_metadata(model_key, spec.raw_config, (3, 64, 64))
    assert "class_names" not in props
    assert props["normalization"] == normalization.name
    assert props["mean"] == ",".join(f"{v:g}" for v in normalization.mean)


def test_stamping_keeps_other_entries(tmp_path: Path) -> None:
    """Re-stamping replaces the suite's keys and keeps foreign ``metadata_props``."""
    path = _export(_classifier(), NSFW, tmp_path)["onnx_fp16"]
    stamp_metadata(path, {"producer_note": "kept"})
    stamp_metadata(path, {"task": "classification"})
    assert read_metadata(path)["producer_note"] == "kept"
    keys = [prop.key for prop in load_onnx().load(str(path)).metadata_props]
    assert keys.count("task") == 1
