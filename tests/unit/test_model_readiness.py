"""Per-format readiness flags of the GUI model catalog (``pytorch_ready`` / ``onnx_*_ready``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.cloud.variations import (
    ONNX_FP16,
    ONNX_FP32,
    PYTORCH_FP32,
    STAGED_FOLDERS,
    variation_dirs,
    variation_ready,
)
from training.export.validation import resolve_model_stem
from training.server.routes.gui_model_stats import model_with_stats

MODEL_KEY = "film_restorer"
INFO = {"name": "Universal Film Restorer", "category": "restoration"}


def _flags(tmp_path: Path) -> dict[str, bool]:
    stats = model_with_stats(MODEL_KEY, INFO, tmp_path, tmp_path / "hub", [])
    return {
        "pytorch_ready": stats.pytorch_ready,
        "onnx_fp32_ready": stats.onnx_fp32_ready,
        "onnx_fp16_ready": stats.onnx_fp16_ready,
    }


def test_no_binaries_means_nothing_ready(tmp_path: Path) -> None:
    """A model without exported files reports every format as not ready."""
    assert not any(_flags(tmp_path).values())


@pytest.mark.parametrize(
    ("relative_dir", "suffix", "flag"),
    [
        ("", ".pt", "pytorch_ready"),
        ("checkpoints", ".pt", "pytorch_ready"),
        ("dist/kaggle/pytorch_fp32", ".pt", "pytorch_ready"),
        ("dist/kaggle/pytorch", ".pt", "pytorch_ready"),
        ("", "_FP32.onnx", "onnx_fp32_ready"),
        ("dist/kaggle/onnx_fp32", "_FP32.onnx", "onnx_fp32_ready"),
        ("", ".onnx", "onnx_fp16_ready"),
        ("checkpoints", ".onnx", "onnx_fp16_ready"),
        ("dist/kaggle/onnx_fp16", ".onnx", "onnx_fp16_ready"),
    ],
)
def test_each_binary_sets_only_its_flag(
    tmp_path: Path, relative_dir: str, suffix: str, flag: str
) -> None:
    """``<Stem>.pt``, ``<Stem>_FP32.onnx`` and ``<Stem>.onnx`` each set exactly one flag."""
    folder = tmp_path / "hub" / MODEL_KEY / relative_dir
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{resolve_model_stem(MODEL_KEY)}{suffix}").write_bytes(b"x")
    flags = _flags(tmp_path)
    assert flags.pop(flag) is True
    assert not any(flags.values())


def test_other_stems_do_not_count(tmp_path: Path) -> None:
    """Binaries of another model in the same folder do not mark this model ready."""
    folder = tmp_path / "hub" / MODEL_KEY
    folder.mkdir(parents=True)
    (folder / "SomethingElse.pt").write_bytes(b"x")
    assert not variation_ready(folder, PYTORCH_FP32, resolve_model_stem(MODEL_KEY))


def test_variation_dirs_cover_download_and_staged_folders(tmp_path: Path) -> None:
    """Readers look in the Kaggle ``payload_dir`` and in the stager's folder name."""
    for variation in (PYTORCH_FP32, ONNX_FP32, ONNX_FP16):
        dirs = variation_dirs(tmp_path, variation)
        dist = tmp_path / "dist" / "kaggle"
        assert dist / variation.payload_dir in dirs
        assert dist / STAGED_FOLDERS[variation.key] in dirs
