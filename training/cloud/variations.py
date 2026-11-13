"""Canonical Kaggle Models Hub variation layout (single source of truth).

Every model is published as one Kaggle Model with four Model Instances ("variations"):

    pytorch/default  checkpoints/ + root files (README.md, metrics.csv)
    pytorch/fp32     <Stem>.pt
    onnx/fp16        <Stem>.onnx (+ <Stem>.onnx.data only when the weights exceed 2 GB)
    onnx/fp32        <Stem>_FP32.onnx + <Stem>_FP32.onnx.data

The backend publisher, the metadata generator and the notebook generator must all import
this module instead of re-declaring handles or file names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal


class VariationKey(StrEnum):
    """Key of one Kaggle Model Instance: the only accepted variation names (API, CLI, GUI).

    The sidecar API exposes these values as an OpenAPI enum, so the GUI type is generated
    from the same list.
    """

    PYTORCH_DEFAULT = "pytorch/default"
    PYTORCH_FP32 = "pytorch/fp32"
    ONNX_FP16 = "onnx/fp16"
    ONNX_FP32 = "onnx/fp32"


ALL_VARIATIONS = "all"
"""Request value selecting every variation."""
VariationChoice = VariationKey | Literal["all"]
"""One variation key, or ``"all"``."""


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
    VariationKey.PYTORCH_DEFAULT.value,
    "pytorch",
    "default",
    "pytorch_default",
    "PyTorch Checkpoints",
)
PYTORCH_FP32 = KaggleVariation(
    VariationKey.PYTORCH_FP32.value, "pytorch", "fp32", "pytorch_fp32", "PyTorch FP32"
)
ONNX_FP16 = KaggleVariation(VariationKey.ONNX_FP16.value, "onnx", "fp16", "onnx_fp16", "ONNX FP16")
ONNX_FP32 = KaggleVariation(VariationKey.ONNX_FP32.value, "onnx", "fp32", "onnx_fp32", "ONNX FP32")

VARIATIONS: tuple[KaggleVariation, ...] = (PYTORCH_DEFAULT, PYTORCH_FP32, ONNX_FP16, ONNX_FP32)
VARIATION_BY_KEY: dict[str, KaggleVariation] = {v.key: v for v in VARIATIONS}


def selected_variations(choice: str) -> list[str]:
    """Variation keys selected by ``choice`` (``"all"`` or one key); ValueError otherwise."""
    if choice == ALL_VARIATIONS:
        return list(VARIATION_BY_KEY)
    if choice not in VARIATION_BY_KEY:
        raise ValueError(
            f"Unknown variation '{choice}'. Allowed: {[ALL_VARIATIONS, *VARIATION_BY_KEY]}"
        )
    return [str(choice)]


CHECKPOINTS_DIR = "checkpoints"
"""Folder of a model hub dir (and of the ``pytorch/default`` payload) holding checkpoints."""
CHECKPOINT_SUFFIX = ".pth"
"""Suffix of the training checkpoints published in ``pytorch/default``."""
DEFAULT_PAYLOAD_ROOT_FILES: tuple[str, ...] = ("README.md", "metrics.csv")
"""Root files published in ``pytorch/default`` next to ``checkpoints/*.pth``."""


def is_default_payload_file(relative: Path) -> bool:
    """Whether ``relative`` (a path inside a model dir) belongs to the ``pytorch/default`` payload.

    The payload is exactly ``checkpoints/*.pth`` plus :data:`DEFAULT_PAYLOAD_ROOT_FILES`:
    no notebooks, no ``dist/``, no ``.pt`` exports and no other history files.
    """
    parts = relative.parts
    if len(parts) == 1:
        return parts[0] in DEFAULT_PAYLOAD_ROOT_FILES
    return len(parts) == 2 and parts[0] == CHECKPOINTS_DIR and relative.suffix == CHECKPOINT_SUFFIX


def default_payload_files(model_dir: Path) -> list[Path]:
    """Relative paths of the ``pytorch/default`` payload files present under ``model_dir``."""
    candidates = [Path(name) for name in DEFAULT_PAYLOAD_ROOT_FILES]
    ckpt_dir = model_dir / CHECKPOINTS_DIR
    if ckpt_dir.is_dir():
        candidates += sorted(
            Path(CHECKPOINTS_DIR) / f.name for f in ckpt_dir.glob(f"*{CHECKPOINT_SUFFIX}")
        )
    return [rel for rel in candidates if (model_dir / rel).is_file()]


PT_SUFFIX = ".pt"
"""``<Stem>.pt``: self-contained PyTorch FP32 model (module object and weights)."""
ONNX_FP32_SUFFIX = "_FP32.onnx"
"""``<Stem>_FP32.onnx``: FP32 ONNX graph; its weights always live in the external sidecar."""
ONNX_FP16_SUFFIX = ".onnx"
"""``<Stem>.onnx``: FP16 ONNX graph; weights embedded unless they exceed the protobuf limit."""
ONNX_SIDECAR_SUFFIX = ".data"
"""Appended to an ONNX file name to name its external weights sidecar (``<file>.onnx.data``)."""


def onnx_sidecar_name(onnx_name: str) -> str:
    """External weights sidecar file name of the ONNX file ``onnx_name``."""
    return f"{onnx_name}{ONNX_SIDECAR_SUFFIX}"


def variation_files(variation: KaggleVariation, stem: str) -> list[str]:
    """Return the binary file names a variation must contain (excluding checkpoints dir).

    The FP16 sidecar is not listed: it exists only for models whose weights exceed the
    protobuf limit (see :mod:`training.export.onnx_weights`).
    """
    if variation is PYTORCH_FP32:
        return [f"{stem}{PT_SUFFIX}"]
    if variation is ONNX_FP16:
        return [f"{stem}{ONNX_FP16_SUFFIX}"]
    if variation is ONNX_FP32:
        fp32_name = f"{stem}{ONNX_FP32_SUFFIX}"
        return [fp32_name, onnx_sidecar_name(fp32_name)]
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
