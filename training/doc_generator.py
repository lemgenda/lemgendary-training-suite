"""Model card and dataset README generation for the LemGendary model hub.

Writes per-model ``README.md`` files (rendered by ``training.model_card``) from the SSOT
registry and training metrics, refreshes dataset manifests and the hub-level README. Run as
``python -m training.doc_generator``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from training.model_card import DEFAULT_HARDWARE, build_model_readme
from training.registry import get_model_registry
from training.utils.lazy_imports import load_hub_readme_generator, load_variation_staging
from training.utils.paths import get_workspace_root

logger = logging.getLogger("lemtrain.docs.doc_generator")

_MANIFEST_HEADING = "## Physical Data Manifest"


def save_readme(path, text):
    """Write ``text`` to ``path`` as UTF-8."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _get_file_count(path):
    if not os.path.isdir(path):
        return "N/A"
    return sum(1 for e in os.scandir(path) if e.is_file())


def _manifest_rows(dataset_path: str) -> list[str]:
    """Return one table row per data folder that holds files."""
    rows = []
    for folder in ["images", "targets", "labels", "masks"]:
        train_count = _get_file_count(os.path.join(dataset_path, folder, "train"))
        val_count = _get_file_count(os.path.join(dataset_path, folder, "val"))

        # Some datasets have no train/val subfolders, just files in the root folder
        # (e.g. CodeFormer targets)
        if train_count == "N/A" and val_count == "N/A":
            root_count = _get_file_count(os.path.join(dataset_path, folder))
            if root_count != "N/A" and root_count > 0:
                rows.append(f"| **{folder}** | {root_count} (total) | N/A |")

        if train_count != "N/A" or val_count != "N/A":
            rows.append(f"| **{folder}** | {train_count} | {val_count} |")
    return rows


def _merge_manifest(existing_content: str, dataset_name: str, manifest_str: str) -> str:
    """Replace the manifest section, or insert it before the footer rule or at the end."""
    if not existing_content:
        return f"# {dataset_name}\n\n{manifest_str}\n"
    if _MANIFEST_HEADING in existing_content:
        parts = existing_content.split(_MANIFEST_HEADING)
        pre = parts[0].rstrip()
        post = parts[1]
        next_header_idx = post.find("\n## ")
        post = post[next_header_idx:] if next_header_idx != -1 else ""
        return f"{pre}\n\n{manifest_str}\n{post}"
    if "\n---" in existing_content:
        parts = existing_content.rsplit("\n---", 1)
        return f"{parts[0].rstrip()}\n\n{manifest_str}\n\n---{parts[1]}"
    return f"{existing_content.rstrip()}\n\n{manifest_str}\n"


def build_dataset_readme(dataset_name, dataset_path):
    """Insert or refresh the physical data manifest in a dataset README; False if no data."""
    readme_path = os.path.join(dataset_path, "README.md")
    existing_content = ""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            existing_content = fh.read()

    rows = _manifest_rows(dataset_path)
    if not rows:
        return False

    manifest_str = "\n".join(
        [_MANIFEST_HEADING, "", "| Folder | Train | Val |", "| :--- | :--- | :--- |", *rows]
    )
    save_readme(readme_path, _merge_manifest(existing_content, dataset_name, manifest_str))
    return True


class ModelDocumentationService:
    """Service generating model documentation and READMEs using SSOT."""

    def __init__(self) -> None:
        self.registry = get_model_registry()

    def generate_readme(
        self,
        model_key: str,
        epochs_trained: int = 0,
        metrics: dict[str, Any] | None = None,
        hardware: str = DEFAULT_HARDWARE,
    ) -> str:
        """Generate model card markdown string."""
        all_specs = self.registry.get_all_models()
        unified_models = {k: v.raw_config for k, v in all_specs.items()}
        return build_model_readme(model_key, unified_models, epochs_trained, metrics, hardware)

    def write_readme(
        self,
        model_key: str,
        output_dir: Path | str | None = None,
        epochs_trained: int = 0,
        metrics: dict[str, Any] | None = None,
        hardware: str = DEFAULT_HARDWARE,
    ) -> Path:
        """Write model README.md to disk."""
        if output_dir:
            dest_dir = Path(output_dir).resolve()
        else:
            dest_dir = (get_workspace_root() / "LemGendaryModels" / model_key).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        content = self.generate_readme(model_key, epochs_trained, metrics, hardware)
        dest_file = dest_dir / "README.md"
        save_readme(str(dest_file), content)
        return dest_file


def _read_metrics(m_csv: str, m_key: str) -> tuple[int, dict[str, Any]]:
    """Return the epochs trained and the last metrics row of a model's ``metrics.csv``."""
    epochs_trained = 0
    metrics: dict[str, Any] = {}
    if os.path.exists(m_csv):
        try:
            with open(m_csv, "r", encoding="utf-8") as cf:
                reader = list(csv.DictReader(cf))
                if reader:
                    last_ep = reader[-1].get("Epoch")
                    if last_ep and last_ep.isdigit():
                        epochs_trained = int(last_ep) + 1
                    else:
                        epochs_trained = len(reader)
                    metrics = reader[-1]
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            logger.warning("Failed reading metrics %s for %s: %s", m_csv, m_key, exc)
    return epochs_trained, metrics


def _write_model_readme(hub_dir: str, m_key: str, unified_models: dict[str, Any]) -> None:
    """Regenerate the hub README of one model from its ``metrics.csv``."""
    m_dir = os.path.join(hub_dir, m_key)
    os.makedirs(m_dir, exist_ok=True)
    m_readme_path = os.path.join(m_dir, "README.md")
    epochs_trained, metrics = _read_metrics(os.path.join(m_dir, "metrics.csv"), m_key)
    content = build_model_readme(m_key, unified_models, epochs_trained, metrics)
    save_readme(m_readme_path, content)
    print(f"[OK] Generated Model README: {m_readme_path}")


def _stage_variations(m_key: str) -> None:
    """Stage the PyTorch and ONNX binary variations of one model."""
    staging_svc = load_variation_staging()()
    st_res = staging_svc.stage_variations(model_key=m_key, validate=True)
    print(
        f"[OK] Staged variations for model: {m_key} "
        f"(staged={len(st_res.staged_files)}, errors={len(st_res.errors)})"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Return the documentation generator argument parser."""
    parser = argparse.ArgumentParser(description="LemGendary Model Documentation Generator")
    parser.add_argument(
        "--all", action="store_true", help="Regenerate READMEs for all registered models"
    )
    parser.add_argument("--model", type=str, help="Regenerate README for a specific model key")
    parser.add_argument(
        "--stage-variations",
        action="store_true",
        help="Stage binary variations (pytorch, onnx_fp32, onnx_fp16) under dist/kaggle/",
    )
    return parser


def main() -> int:
    """CLI: regenerate model READMEs (and optionally stage variations), then the hub README."""
    parser = _build_parser()
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, "unified_models_v2.yaml")
    hub_dir = os.path.abspath(os.path.join(base_dir, "..", "LemGendaryModels"))

    with open(yaml_path, "r", encoding="utf-8") as f:
        unified_models = yaml.safe_load(f)

    if args.all:
        models_to_process = [k for k in unified_models if k != "_registry_metadata"]
    elif args.model:
        if args.model not in unified_models:
            print(f"[ERROR] Model '{args.model}' not found in unified_models_v2.yaml.")
            return 1
        models_to_process = [args.model]
    else:
        parser.print_help()
        return 0

    for m_key in models_to_process:
        _write_model_readme(hub_dir, m_key, unified_models)
        if args.stage_variations:
            _stage_variations(m_key)

    load_hub_readme_generator()(base_dir)
    print("\n[SUCCESS] Model README Matrix Synchronized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
