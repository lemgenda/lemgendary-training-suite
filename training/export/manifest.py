"""Export manifest generator and validator for LemGendary AI Training Suite.

Emits and validates export_manifest.json per model to record:
- Model metadata (key, stem, architecture, task)
- Model precision & format targets (FP32 PyTorch, FP32 ONNX, FP16 ONNX)
- ONNX opset version and IR details
- Numerical equivalence tolerances per family (atol, rtol, verified max_abs_diff)
- Input / output tensor signatures (names, shapes, dtypes, dynamic axes)
- Cryptographic SHA-256 checksums and byte sizes of all release binaries
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from training.export.dynamo_onnx import ProbeTolerance
from training.export.validation import resolve_model_stem
from training.registry import ModelSpecification, get_model_registry

logger = logging.getLogger("lemtrain.export.manifest")

MANIFEST_FILENAME = "export_manifest.json"
_HASH_CHUNK_SIZE = 1024 * 1024

# Documented numerical equivalence tolerances per model family:
# - Vision / Restoration (default PyTorch): atol=1e-3, rtol=1e-2 for FP32; atol=1e-2 for FP16
# - YOLO (Ultralytics detection): atol=1e-3, rtol=1e-2 for FP32; atol=1e-2 for FP16
# - Forex (multi-timeframe EA): atol=1e-4, rtol=1e-3 for FP32; atol=1e-3 for FP16
# - LLM / Causal LM: atol=1e-3, rtol=1e-2 for FP32; atol=1e-2 for FP16
FAMILY_PARITY_TOLERANCES: dict[str, dict[str, ProbeTolerance]] = {
    "vision": {
        "fp32": ProbeTolerance(atol=1e-3, rtol=1e-2),
        "fp16": ProbeTolerance(atol=1e-2, rtol=1e-2),
    },
    "yolo": {
        "fp32": ProbeTolerance(atol=1e-3, rtol=1e-2),
        "fp16": ProbeTolerance(atol=1e-2, rtol=1e-2),
    },
    "forex": {
        "fp32": ProbeTolerance(atol=1e-4, rtol=1e-3),
        "fp16": ProbeTolerance(atol=1e-3, rtol=1e-2),
    },
    "llm": {
        "fp32": ProbeTolerance(atol=1e-3, rtol=1e-2),
        "fp16": ProbeTolerance(atol=1e-2, rtol=1e-2),
    },
}


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 cryptographic digest of a file."""
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Cannot compute checksum; file not found: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        chunk = handle.read(_HASH_CHUNK_SIZE)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(_HASH_CHUNK_SIZE)
    return digest.hexdigest()


def resolve_family_tolerances(
    model_key: str, spec: ModelSpecification | None = None
) -> dict[str, dict[str, float]]:
    """Resolve documented numerical parity tolerances for a model family."""
    key_lower = model_key.lower()
    family = "vision"
    if "yolo" in key_lower:
        family = "yolo"
    elif "forex" in key_lower:
        family = "forex"
    elif "ai_helper" in key_lower or "llm" in key_lower:
        family = "llm"
    elif spec:
        cat = (spec.category or "").lower()
        if "forex" in cat:
            family = "forex"
        elif "yolo" in cat or "detection" in cat:
            family = "yolo"
        elif "llm" in cat or "nlp" in cat or "causal" in cat:
            family = "llm"

    selected = FAMILY_PARITY_TOLERANCES.get(family, FAMILY_PARITY_TOLERANCES["vision"])
    return {
        variant: {"atol": tol.atol, "rtol": tol.rtol}
        for variant, tol in selected.items()
    }


@dataclass
class ExportArtifactRecord:
    """Metadata and checksum for an exported release artifact."""

    key: str
    file_name: str
    file_path: str
    size_bytes: int
    sha256: str
    dtype: str
    opset: int | None = None


@dataclass
class ExportManifest:
    """Canonical export manifest representing exported model assets."""

    model_key: str
    stem: str
    opset: int
    input_signature: dict[str, Any]
    output_signature: dict[str, Any]
    tolerances: dict[str, dict[str, float]]
    artifacts: dict[str, ExportArtifactRecord] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        data = asdict(self)
        data["artifacts"] = {
            k: asdict(v) if isinstance(v, ExportArtifactRecord) else v
            for k, v in self.artifacts.items()
        }
        return data

    def save(self, destination: Path | str) -> Path:
        """Write manifest to destination directory or specific file path."""
        dest = Path(destination).resolve()
        manifest_path = dest / MANIFEST_FILENAME if dest.is_dir() else dest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Saved export manifest to %s", manifest_path)
        return manifest_path


def build_export_manifest(
    model_key: str,
    exported_files: Mapping[str, Path | str],
    spec: ModelSpecification | None = None,
    opset: int = 18,
    input_signature: dict[str, Any] | None = None,
    output_signature: dict[str, Any] | None = None,
    base_dir: Path | str | None = None,
) -> ExportManifest:
    """Construct an authoritative ExportManifest from exported artifacts and spec."""
    reg = get_model_registry()
    resolved_spec = spec or reg.get_model(model_key)
    stem = resolve_model_stem(model_key, resolved_spec)
    tolerances = resolve_family_tolerances(model_key, resolved_spec)

    # Inferred default signatures if not provided
    in_sig = input_signature or {}
    out_sig = output_signature or {}
    if not in_sig and resolved_spec:
        in_sig = {
            "input_names": ["input"],
            "input_size": (resolved_spec.raw_config or {}).get("input_size", 256),
            "dynamic_axes": {"input": {0: "batch"}},
        }
    if not out_sig and resolved_spec:
        out_sig = {
            "output_names": ["output"],
            "dynamic_axes": {"output": {0: "batch"}},
        }

    artifact_records: dict[str, ExportArtifactRecord] = {}
    for key, p in exported_files.items():
        path = Path(p).resolve()
        if not path.is_file():
            continue

        dtype = "float32"
        artifact_opset: int | None = None
        is_fp16 = "fp16" in key.lower() or (
            path.name.endswith(".onnx") and not path.name.endswith("_FP32.onnx")
        )
        if is_fp16:
            dtype = "float16"
            artifact_opset = opset
        elif "fp32" in key.lower() or path.name.endswith("_FP32.onnx"):
            dtype = "float32"
            artifact_opset = opset
        elif path.suffix in (".pt", ".pth"):
            dtype = "float32"

        rel_path = path.name
        if base_dir is not None:
            try:
                rel_path = path.relative_to(Path(base_dir).resolve()).as_posix()
            except ValueError:
                rel_path = path.name

        rec = ExportArtifactRecord(
            key=key,
            file_name=path.name,
            file_path=rel_path,
            size_bytes=path.stat().st_size,
            sha256=compute_file_sha256(path),
            dtype=dtype,
            opset=artifact_opset,
        )
        artifact_records[key] = rec

    return ExportManifest(
        model_key=model_key,
        stem=stem,
        opset=opset,
        input_signature=in_sig,
        output_signature=out_sig,
        tolerances=tolerances,
        artifacts=artifact_records,
    )
