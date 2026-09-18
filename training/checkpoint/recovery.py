"""BFS checkpoint recovery engine across Kaggle mounts, model hubs, and local scratch dirs."""

import logging
from pathlib import Path
import re
import shutil
from typing import Any

from training.utils.paths import get_project_root, get_workspace_root

logger = logging.getLogger("lemtrain.checkpoint.recovery")


class CheckpointRecoveryEngine:
    """Discovers and synchronizes model checkpoints across distributed cloud and local filesystems."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        project_root: Path | None = None,
        env: str = "local",
    ) -> None:
        self.workspace_root = workspace_root or get_workspace_root()
        self.project_root = project_root or get_project_root()
        self.env = env

    def find_candidate_roots(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Compute all potential search roots containing checkpoints for the target model."""
        cfg = config or {}
        roots: list[Path] = []

        # Local Hub & Checkpoints
        p_paths = cfg.get("paths", {})
        export_root = p_paths.get("export_root", "../LemGendaryModels")
        hub_ckpt = (self.workspace_root / export_root / model_name / "checkpoints").resolve()
        if hub_ckpt.exists():
            roots.append(hub_ckpt)

        local_ckpt = self.project_root / "checkpoints"
        if local_ckpt.exists():
            roots.append(local_ckpt)

        scratch_ckpt = self.project_root / "_local_checkpoints" / model_name
        if scratch_ckpt.exists():
            roots.append(scratch_ckpt)

        # Kaggle input mounts and working directory
        if self.env == "kaggle" or Path("/kaggle/input").exists():
            kaggle_working = Path("/kaggle/working/LemGendaryModels") / model_name / "checkpoints"
            if kaggle_working.exists():
                roots.append(kaggle_working)

            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                model_norm = model_name.lower().replace("_", "").replace("-", "")
                try:
                    for owner in kaggle_input.iterdir():
                        if not owner.is_dir():
                            continue
                        for mount in owner.iterdir():
                            if not mount.is_dir():
                                continue
                            mount_norm = mount.name.lower().replace("_", "").replace("-", "")
                            if model_norm in mount_norm or "checkpoint" in mount_norm:
                                roots.append(mount)
                except OSError as exc:
                    logger.debug("Failed traversing /kaggle/input: %s", exc)

            # Attempt kagglehub download probe
            try:
                import importlib
                kh_mod = importlib.import_module("kagglehub")
                k_user = cfg.get("kaggle_username", "lemtreursi")
                k_slug = model_name.replace("_", "-")
                if "nima-aesthetic" in k_slug:
                    k_slug = k_slug.replace("nima-aesthetic", "nima-aesthetics")
                prefix = cfg.get("kaggle_slug_prefix", "lemgendary-")
                suffix = cfg.get("kaggle_slug_suffix", "-checkpoints")
                k_handle = f"{k_user}/{prefix}{k_slug}{suffix}/pytorch/default"

                model_download_fn = getattr(kh_mod, "model_download", None)
                if callable(model_download_fn):
                    dl_path = Path(model_download_fn(k_handle))
                    if dl_path.exists():
                        logger.info("KaggleHub resolved checkpoint directory: %s", dl_path)
                        roots.insert(0, dl_path)
            except Exception as kh_exc:
                logger.debug("KaggleHub checkpoint probe notice: %s", kh_exc)

        # Remove duplicate roots
        unique_roots: list[Path] = []
        for r in roots:
            resolved = r.resolve()
            if resolved not in unique_roots:
                unique_roots.append(resolved)
        return unique_roots

    def discover_checkpoints(
        self,
        candidate_roots: list[Path],
        model_name: str,
    ) -> dict[str, Path]:
        """Index available checkpoints for the target model across candidate roots."""
        discovered: dict[str, Path] = {}
        epoch_ckpts: list[tuple[int, Path]] = []

        for root in candidate_roots:
            if not root.exists():
                continue

            # BFS search up to depth 4
            queue: list[tuple[Path, int]] = [(root, 0)]
            while queue:
                curr_dir, depth = queue.pop(0)
                if depth > 4:
                    continue
                try:
                    entries = list(curr_dir.iterdir())
                except OSError:
                    continue

                for entry in entries:
                    if entry.is_dir():
                        queue.append((entry, depth + 1))
                        continue

                    if entry.is_file() and entry.suffix == ".pth":
                        name_lower = entry.name.lower()
                        if "latest" in name_lower and "latest" not in discovered:
                            discovered["latest"] = entry
                        elif "best" in name_lower and "best" not in discovered:
                            discovered["best"] = entry
                        elif "progress" in name_lower and "progress" not in discovered:
                            discovered["progress"] = entry

                        # Check for epoch patterns (e.g. model_epoch_015.pth)
                        match = re.search(r"epoch_?(\d+)", name_lower)
                        if match:
                            epoch_ckpts.append((int(match.group(1)), entry))

        # Sort epoch checkpoints by descending epoch
        epoch_ckpts.sort(key=lambda x: x[0], reverse=True)
        if epoch_ckpts:
            discovered["latest_epoch"] = epoch_ckpts[0][1]

        return discovered

    def sync_to_local_hub(
        self,
        source_root: Path,
        target_dir: Path,
        model_name: str,
    ) -> list[Path]:
        """Synchronize checkpoint files and metrics.csv from source into destination."""
        target_dir.mkdir(parents=True, exist_ok=True)
        synced: list[Path] = []

        if not source_root.exists():
            return synced

        # Sync metrics.csv
        for candidate_csv in ["metrics.csv", f"{model_name}_metrics.csv"]:
            src_csv = source_root / candidate_csv
            if src_csv.exists():
                dst_csv = target_dir / "metrics.csv"
                try:
                    shutil.copy2(src_csv, dst_csv)
                    synced.append(dst_csv)
                    logger.info("Synced metrics log: %s -> %s", src_csv, dst_csv)
                except OSError as exc:
                    logger.warning("Failed syncing metrics.csv: %s", exc)

        # Sync .pth files
        for pth in source_root.glob("**/*.pth"):
            if pth.is_file():
                dst_pth = target_dir / pth.name
                try:
                    shutil.copy2(pth, dst_pth)
                    synced.append(dst_pth)
                    logger.info("Synced checkpoint: %s -> %s", pth.name, dst_pth)
                except OSError as exc:
                    logger.warning("Failed syncing checkpoint %s: %s", pth, exc)

        return synced
