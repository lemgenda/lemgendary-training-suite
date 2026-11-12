"""Canonical Kaggle Models Hub variation layout (single source of truth).

Every model is published as one Kaggle Model with four Model Instances ("variations"):

    pytorch/default  checkpoints/ + root files (README.md, metrics.csv)
    pytorch/fp32     <Stem>.pt
    onnx/fp16        <Stem>.onnx
    onnx/fp32        <Stem>_FP32.onnx + <Stem>_FP32.onnx.data

The backend publisher, the metadata generator and the notebook generator must all import
this module instead of re-declaring handles or file names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KaggleVariation:
    """One Kaggle Model Instance of a LemGendary model."""

    key: str
    framework: str
    instance_slug: str
    payload_dir: str
    label: str

    def handle(self, owner: str, model_slug: str) -> str:
        """Return the kagglehub / Kaggle API handle for this variation."""
        return f"{owner}/{model_slug}/{self.framework}/{self.instance_slug}"


PYTORCH_DEFAULT = KaggleVariation(
    "pytorch/default", "pytorch", "default", "pytorch_default", "PyTorch Checkpoints"
)
PYTORCH_FP32 = KaggleVariation("pytorch/fp32", "pytorch", "fp32", "pytorch_fp32", "PyTorch FP32")
ONNX_FP16 = KaggleVariation("onnx/fp16", "onnx", "fp16", "onnx_fp16", "ONNX FP16")
ONNX_FP32 = KaggleVariation("onnx/fp32", "onnx", "fp32", "onnx_fp32", "ONNX FP32")

VARIATIONS: tuple[KaggleVariation, ...] = (PYTORCH_DEFAULT, PYTORCH_FP32, ONNX_FP16, ONNX_FP32)
VARIATION_BY_KEY: dict[str, KaggleVariation] = {v.key: v for v in VARIATIONS}


def variation_files(variation: KaggleVariation, stem: str) -> list[str]:
    """Return the binary file names a variation must contain (excluding checkpoints dir)."""
    if variation is PYTORCH_FP32:
        return [f"{stem}.pt"]
    if variation is ONNX_FP16:
        return [f"{stem}.onnx"]
    if variation is ONNX_FP32:
        return [f"{stem}_FP32.onnx", f"{stem}_FP32.onnx.data"]
    return []


STAGED_FOLDERS: dict[str, str] = {
    PYTORCH_FP32.key: "pytorch",
    ONNX_FP32.key: "onnx_fp32",
    ONNX_FP16.key: "onnx_fp16",
}
"""Variation key -> ``dist/kaggle`` folder the export stager and recovery sync write to.

Kaggle downloads use ``payload_dir`` instead; readers check both.
"""


def variation_dirs(hub_dir: Path, variation: KaggleVariation) -> list[Path]:
    """Folders of a model hub dir that may hold ``variation``'s binaries.

    The hub root, its ``checkpoints/`` folder and the variation's ``dist/kaggle``
    folders (the downloaded ``payload_dir`` and the staged folder name).
    """
    dist = hub_dir / "dist" / "kaggle"
    dirs = [hub_dir, hub_dir / "checkpoints", dist / variation.payload_dir]
    staged = STAGED_FOLDERS.get(variation.key)
    if staged and staged != variation.payload_dir:
        dirs.append(dist / staged)
    return dirs


def variation_ready(hub_dir: Path, variation: KaggleVariation, stem: str) -> bool:
    """Whether ``variation``'s main binary (``variation_files``' first) exists for ``stem``."""
    files = variation_files(variation, stem)
    if not files:
        return False
    return any((folder / files[0]).is_file() for folder in variation_dirs(hub_dir, variation))
