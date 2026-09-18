"""Common notebook builder serialization and cross-directory synchronization.

Handles JSON encoding, validation, atomic file persistence, and automatic distribution
to sibling LemGendaryDatasets manifolds and workspace training directories.
"""

import json
import os
from pathlib import Path
from typing import Any

def _get_project_root() -> Path:
    """Resolve project root directory across training suite or datasets repositories."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "config.yaml").exists() or (parent / "unified_data.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here.parent.parent.parent


def write_notebook(content: dict[str, Any], output_path: str | Path) -> str | None:
    """Validate and write a notebook content dictionary to disk as UTF-8 JSON.

    Returns the formatted JSON string if successful, or None on failure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_str = json.dumps(content, indent=4)
        json.loads(json_str)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        return json_str
    except Exception as e:
        print(f"[ERROR] JSON Validation failed for {out_path.name}: {e}")
        print("[REMEDY] This usually means the generated notebook syntax is invalid. Check 'unified_models.yaml' for trailing commas or malformed strings.")
        return None


def sync_manifold_notebooks(
    model_key: str,
    json_str: str,
    filename: str,
    unified_models_registry: dict[str, Any] | None = None,
) -> None:
    """Sync generated notebook JSON to matching LemGendaryDatasets manifold directories."""
    project_root = _get_project_root()
    datasets_hub_root = os.path.abspath(os.path.join(project_root, "../LemGendaryDatasets"))

    if not unified_models_registry:
        return

    m_info = unified_models_registry.get(model_key, {})
    ds_raw = m_info.get("datasets", []) or m_info.get("dataset", [])
    if isinstance(ds_raw, str):
        ds_list = [ds_raw]
    elif isinstance(ds_raw, (list, tuple)):
        ds_list = list(ds_raw)
    else:
        ds_list = []

    if model_key == "professional_multitask_restoration":
        target_candidates = ["LemGendizedProfessionalMultitaskRestorationLarge", "professional_multitask_restoration"]
    else:
        target_candidates = list(ds_list)
        if model_key not in target_candidates:
            target_candidates.append(model_key)

    synced_dirs: set[str] = set()
    for target_folder in target_candidates:
        if not target_folder:
            continue
        clean_name = target_folder
        if "_" in clean_name or "-" in clean_name:
            pascal_name = "".join(part.capitalize() for part in clean_name.replace("-", "_").split("_"))
        else:
            pascal_name = clean_name

        possible_manifold_folders = [
            target_folder,
            f"{target_folder}Large",
            f"LemGendized{pascal_name}",
            f"LemGendized{pascal_name}Large",
            f"LemGendized{target_folder}Large",
            f"LemGendized{target_folder}",
        ]

        for m_folder in possible_manifold_folders:
            ds_dir = os.path.join(datasets_hub_root, m_folder)
            if os.path.exists(ds_dir) and ds_dir not in synced_dirs:
                synced_dirs.add(ds_dir)
                ds_output_path = os.path.join(ds_dir, filename)
                try:
                    with open(ds_output_path, "w", encoding="utf-8") as f:
                        f.write(json_str)
                    print(f"[OK] Synchronized Dataset Manifold Notebook: {ds_output_path}")
                except OSError:
                    pass


def sync_workspace_training_notebook(
    subfolder_name: str,
    filename: str,
    json_str: str,
    display_title: str,
) -> None:
    """Sync generated notebook JSON to workspace training folder (e.g. kaggle_training)."""
    project_root = _get_project_root()
    workspace_root = os.path.abspath(os.path.join(project_root, ".."))
    target_dir = os.path.join(workspace_root, subfolder_name)
    os.makedirs(target_dir, exist_ok=True)
    out_file = os.path.join(target_dir, filename)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[OK] Synchronized {display_title} Training Notebook: {out_file}")
    except OSError:
        pass
