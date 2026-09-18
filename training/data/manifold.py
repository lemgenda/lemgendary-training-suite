"""Canonical Manifold Resolver for LemGendary Model Training Suite."""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any
import yaml

from training.utils.paths import get_project_root, get_workspace_root

logger = logging.getLogger("lemtrain.manifold")


@dataclass(frozen=True)
class ManifoldInfo:
    """Structured inspection summary for a resolved dataset manifold."""

    name: str
    path: Path
    task_type: str = "restoration"
    container_type: str = "directory"
    images_dir: Path | None = None
    targets_dir: Path | None = None
    masks_dir: Path | None = None
    labels_dir: Path | None = None
    metadata_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ManifoldResolver:
    """Resolves dataset manifold locations and container layouts across local, Colab, and Kaggle environments."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        project_root: Path | None = None,
        env: str = "local",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.workspace_root = workspace_root or get_workspace_root()
        self.project_root = project_root or get_project_root()
        self.env = env
        self.config = config or {}

    def get_search_roots(self) -> list[Path]:
        """Compute search directories for dataset manifolds in order of priority."""
        roots: list[Path] = []

        cfg_paths = self.config.get("paths", {})
        custom_root = cfg_paths.get("datasets_root") or self.config.get("datasets_dir")
        if custom_root:
            p = Path(custom_root)
            if not p.is_absolute():
                p = self.workspace_root / p
            roots.append(p)

        if self.env == "kaggle":
            roots.extend([
                Path("/kaggle/working/LemGendaryDatasets"),
                Path("/kaggle/input"),
            ])
        elif self.env == "colab":
            roots.extend([
                Path("/content/LemGendaryDatasets"),
                Path("/content/drive/MyDrive/LemGendaryDatasets"),
            ])

        roots.extend([
            self.workspace_root / "LemGendaryDatasets",
            self.project_root / "data" / "datasets",
            self.project_root / "data" / "forex",
            self.project_root / "data",
            self.workspace_root / "lemgendary-datasets" / "manifolds",
            self.workspace_root / "lemgendary-datasets" / "data",
        ])

        # De-duplicate while preserving order
        unique_roots: list[Path] = []
        for r in roots:
            resolved = r.resolve()
            if resolved not in unique_roots:
                unique_roots.append(resolved)
        return unique_roots

    def generate_name_candidates(self, manifold_name: str) -> list[str]:
        """Generate common name variations (prefixes, suffixes) for fuzzy resolution."""
        candidates: list[str] = [manifold_name]

        suffix = "KaggleReady" if self.env == "kaggle" else ""
        if suffix and not manifold_name.endswith(suffix):
            candidates.append(f"{manifold_name}{suffix}")

        if not manifold_name.lower().startswith("lemgendized"):
            candidates.append(f"LemGendized{manifold_name}")
            if suffix:
                candidates.append(f"LemGendized{manifold_name}{suffix}")

        return candidates

    def resolve_manifold(self, manifold_name: str) -> Path | None:
        """Locate a dataset manifold directory on disk."""
        candidates = self.generate_name_candidates(manifold_name)
        search_roots = self.get_search_roots()

        # Phase A: Direct path lookup in search roots
        for root in search_roots:
            if not root.exists():
                continue
            for cand in candidates:
                direct = root / cand
                if direct.exists() and (direct.is_dir() or direct.suffix in {".parquet", ".tar"}):
                    return direct

                # Subdirectory check (e.g. root / cand / "forex")
                sub = direct / "forex"
                if sub.exists() and sub.is_dir():
                    return sub

                # Case-insensitive direct match in root
                try:
                    for child in root.iterdir():
                        if child.name.lower() == cand.lower():
                            return child
                except OSError as exc:
                    logger.debug("Failed listing directory %s: %s", root, exc)

        # Phase B: Kaggle /kaggle/input BFS traversal (depth <= 3)
        if self.env == "kaggle" or any("/kaggle/input" in str(r).replace("\\", "/") for r in search_roots):
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                simplified_target = manifold_name.lower().replace("-", "").replace("_", "").replace("lemgendized", "")
                for suf in ["kaggleready", "large", "mini"]:
                    simplified_target = simplified_target.replace(suf, "")

                queue: list[tuple[Path, int]] = [(kaggle_input, 0)]
                while queue:
                    curr_dir, depth = queue.pop(0)
                    if depth > 3:
                        continue
                    try:
                        children = list(curr_dir.iterdir())
                    except OSError:
                        continue

                    for child in children:
                        if not child.is_dir():
                            continue
                        name_norm = child.name.lower().replace("-", "").replace("_", "").replace("lemgendized", "")
                        if simplified_target and simplified_target in name_norm:
                            if (child / "images").exists() or (child / "targets").exists() or (child / "forex").exists():
                                return child
                            # Check single nested subfolder
                            try:
                                for sub in child.iterdir():
                                    if sub.is_dir() and ((sub / "images").exists() or (sub / "targets").exists()):
                                        return sub
                            except OSError:
                                pass
                        queue.append((child, depth + 1))

        return None

    def inspect_manifold(self, path_or_name: str | Path) -> ManifoldInfo:
        """Inspect and parse structured metadata for a resolved manifold."""
        if isinstance(path_or_name, str):
            resolved = self.resolve_manifold(path_or_name)
            if resolved is None:
                raise FileNotFoundError(f"Manifold '{path_or_name}' could not be resolved across search roots.")
            path = resolved
            name = path_or_name
        else:
            path = path_or_name
            name = path.name

        if not path.exists():
            raise FileNotFoundError(f"Manifold path does not exist: {path}")

        meta: dict[str, Any] = {}
        meta_file: Path | None = None
        for candidate_meta in ["dataset_info.yaml", "metadata.json", "index.json"]:
            mf = path / candidate_meta
            if mf.exists():
                meta_file = mf
                try:
                    if candidate_meta.endswith(".yaml") or candidate_meta.endswith(".yml"):
                        meta = yaml.safe_load(mf.read_text(encoding="utf-8")) or {}
                    else:
                        import json
                        meta = json.loads(mf.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed parsing metadata file %s: %s", mf, exc)
                break

        # Container type detection
        container_type = "directory"
        if "container" in meta and isinstance(meta["container"], dict):
            container_type = meta["container"].get("primary", "directory")
        elif any(f.suffix == ".parquet" for f in path.iterdir() if f.is_file()):
            container_type = "parquet"
        elif any(f.suffix == ".tar" for f in path.iterdir() if f.is_file()):
            container_type = "webdataset"
        elif (path / "index.json").exists() and not (path / "images").exists():
            container_type = "mds"

        # Task type detection
        task_type = meta.get("task_type", "")
        if not task_type:
            if container_type == "parquet" or "forex" in name.lower() or (path / "forex").exists():
                task_type = "forex"
            elif (path / "masks").exists():
                task_type = "segmentation"
            elif (path / "labels").exists():
                task_type = "face_detection"
            else:
                task_type = "restoration"

        images_dir = path / "images" if (path / "images").exists() else None
        targets_dir = path / "targets" if (path / "targets").exists() else None
        masks_dir = path / "masks" if (path / "masks").exists() else None
        labels_dir = path / "labels" if (path / "labels").exists() else None

        return ManifoldInfo(
            name=name,
            path=path,
            task_type=task_type,
            container_type=container_type,
            images_dir=images_dir,
            targets_dir=targets_dir,
            masks_dir=masks_dir,
            labels_dir=labels_dir,
            metadata_file=meta_file,
            metadata=meta,
        )

    def resolve_split_path(
        self,
        manifold_path: Path,
        folder_name: str,
        split: str,
        filename: str,
        ext: str = "",
    ) -> Path | None:
        """Resolve individual sample asset across canonical split directory structures."""
        base_name = Path(filename).stem if ext else filename
        suffix = ext if ext else ""
        target_name = f"{base_name}{suffix}"

        # 1. Primary path: <manifold>/<folder>/<split>/<target_name>
        primary = manifold_path / folder_name / split / target_name
        if primary.exists():
            return primary

        # 2. Flat split path: <manifold>/<split>/<folder>/<target_name>
        flat_split = manifold_path / split / folder_name / target_name
        if flat_split.exists():
            return flat_split

        # 3. Direct folder path: <manifold>/<folder>/<target_name>
        direct = manifold_path / folder_name / target_name
        if direct.exists():
            return direct

        # 4. Fallback search across alternate extensions if ext was empty
        if not ext:
            for alt_ext in [".png", ".jpg", ".jpeg", ".webp"]:
                alt_path = manifold_path / folder_name / split / f"{base_name}{alt_ext}"
                if alt_path.exists():
                    return alt_path

        return None
