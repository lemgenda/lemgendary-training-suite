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

        # Suffix stripping / normalization: e.g. LemGendizedNimaAestheticLarge -> LemGendizedNimaAesthetic
        if manifold_name.endswith("Large"):
            stripped = manifold_name[:-5]
            if stripped not in candidates:
                candidates.append(stripped)
        elif not manifold_name.endswith("Large") and not manifold_name.endswith("KaggleReady"):
            legacy = f"{manifold_name}Large"
            if legacy not in candidates:
                candidates.append(legacy)

        suffix = "KaggleReady" if self.env == "kaggle" else ""
        if suffix and not manifold_name.endswith(suffix):
            candidates.append(f"{manifold_name}{suffix}")

        if not manifold_name.lower().startswith("lemgendized"):
            candidates.append(f"LemGendized{manifold_name}")
            if suffix:
                candidates.append(f"LemGendized{manifold_name}{suffix}")

        for c in list(candidates):
            if c.endswith("Large"):
                s = c[:-5]
                if s not in candidates:
                    candidates.append(s)

        return candidates

    @staticmethod
    def is_valid_manifold_dir(p: Path) -> bool:
        """Check if directory contains dataset images, targets, shards, parquet, or manifest metadata."""
        if not p.is_dir():
            return False
        for sub in ("images", "targets", "masks", "forex", "shards", "train", "val", "labels", "mds", "litdata", "parquet"):
            if (p / sub).exists():
                return True
        for meta in ("dataset_info.yaml", "dataset-metadata.json", "index.json", "classes.txt", "category.txt"):
            if (p / meta).exists():
                return True
        try:
            for item in p.iterdir():
                if item.suffix.lower() in (".parquet", ".tar", ".webp", ".jpg", ".png", ".jpeg"):
                    return True
        except OSError:
            pass
        return False

    @classmethod
    def scan_kaggle_input_manifolds(cls, root: Path = Path("/kaggle/input"), max_depth: int = 4) -> list[Path]:
        """Discover all attached dataset manifolds in /kaggle/input regardless of name."""
        if not root.exists() or not root.is_dir():
            return []
        discovered: list[Path] = []
        visited: set[Path] = set()
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue:
            curr, depth = queue.pop(0)
            if depth > max_depth or not curr.is_dir():
                continue
            if curr != root and cls.is_valid_manifold_dir(curr):
                real_p = curr.resolve()
                if real_p not in visited:
                    visited.add(real_p)
                    discovered.append(curr)
                continue
            try:
                children = list(curr.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_dir():
                    if child.name.lower() == "models" and curr == root:
                        continue
                    if cls.is_valid_manifold_dir(child):
                        real_p = child.resolve()
                        if real_p not in visited:
                            visited.add(real_p)
                            discovered.append(child)
                    else:
                        queue.append((child, depth + 1))
        return discovered

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

                sub = direct / "forex"
                if sub.exists() and sub.is_dir():
                    return sub

                try:
                    for child in root.iterdir():
                        if child.name.lower() == cand.lower():
                            return child
                except OSError as exc:
                    logger.debug("Failed listing directory %s: %s", root, exc)

        # Phase B: Kaggle /kaggle/input inspection (train on whichever dataset or datasets are attached)
        if self.env == "kaggle" or any("/kaggle/input" in str(r).replace("\\", "/") for r in search_roots) or Path("/kaggle/input").exists():
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                attached_manifolds = self.scan_kaggle_input_manifolds(kaggle_input)
                if attached_manifolds:
                    simplified_target = manifold_name.lower().replace("-", "").replace("_", "").replace("lemgendized", "")
                    for suf in ["kaggleready", "large", "mini"]:
                        simplified_target = simplified_target.replace(suf, "")

                    # 1. Exact or candidate name match
                    for m in attached_manifolds:
                        if m.name.lower() in [c.lower() for c in candidates]:
                            return m

                    # 2. Fuzzy name match against manifold or parent dataset slug
                    for m in attached_manifolds:
                        name_norm = m.name.lower().replace("-", "").replace("_", "").replace("lemgendized", "")
                        parent_norm = m.parent.name.lower().replace("-", "").replace("_", "").replace("lemgendized", "")
                        if simplified_target and (simplified_target in name_norm or simplified_target in parent_norm):
                            return m

                    # 3. Dynamic Fallback: Auto-bind attached Kaggle dataset manifold
                    fallback_m = attached_manifolds[0]
                    logger.info(
                        "Auto-bound attached Kaggle dataset manifold '%s' for requested '%s'",
                        fallback_m,
                        manifold_name,
                    )
                    return fallback_m

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
        elif meta.get("canonical_format"):
            container_type = meta.get("canonical_format")
        elif meta.get("format"):
            container_type = meta.get("format")
        elif (path / "shards").exists() or any(f.suffix == ".tar" for f in path.iterdir() if f.is_file()):
            container_type = "webdataset"
        elif (path / "mds").exists() or ((path / "index.json").exists() and not (path / "images").exists()):
            container_type = "mds"
        elif (path / "litdata").exists():
            container_type = "litdata"
        elif any(f.suffix == ".parquet" for f in path.iterdir() if f.is_file()):
            container_type = "parquet"

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
