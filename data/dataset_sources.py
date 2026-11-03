"""Dataset discovery for MultiTaskDataset.

Split directories, dataset path resolution, manifest loading and sample-fraction
subsetting.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import zipfile

from training.data.manifold import ManifoldResolver

logger = logging.getLogger("lemtrain.dataset")

_IMAGE_EXTS = (".jpg", ".png", ".jpeg", ".webp")
# Sub-folders that mark a dataset directory as usable, per environment.
_KAGGLE_LAYOUT = ("images", "targets", "shards")
_LOCAL_LAYOUT = ("images", "targets")


def _print_kaggle_mounts():
    """Debug listing of the datasets mounted in ``/kaggle/input`` (two levels deep)."""
    print("[SYNC] [DEBUG] Mounted datasets in /kaggle/input:")
    try:
        if os.path.exists("/kaggle/input"):
            for d in os.listdir("/kaggle/input"):
                d_path = os.path.join("/kaggle/input", d)
                if os.path.isdir(d_path):
                    _print_mount(d, d_path)
        else:
            print("  -> /kaggle/input does not exist!")
    except OSError as list_err:
        print("  -> Error listing /kaggle/input:", list_err)


def _print_mount(d, d_path):
    """Print one mounted dataset and the first entries of each of its sub-folders."""
    print(f"  -> {d} contains: {os.listdir(d_path)[:10]}")
    for sub in os.listdir(d_path):
        sub_path = os.path.join(d_path, sub)
        if os.path.isdir(sub_path):
            try:
                print(f"      -> {sub} contains: {os.listdir(sub_path)[:5]}")
            except OSError as sub_err:
                logger.debug("Failed listing %s: %s", sub_path, sub_err)


def _dataset_suffix(env, config):
    """Dataset-name suffix for the environment and execution mode."""
    if env == "kaggle":
        return "KaggleReady"
    return (
        config.get("execution", {})
        .get("suffixes", {})
        .get(config.get("execution", {}).get("mode", "training"), "")
    )


def _dataset_names(raw_dataset_names, suffix):
    """Expand registry names with the suffixed and ``LemGendized`` variants, in lookup order."""
    # 2026 Resilience: auto-add the 'LemGendized' prefix; guard against double-suffixing
    dataset_names = []
    for name in raw_dataset_names:
        dataset_names.append(name)
        ds_entry = name if (suffix and name.endswith(suffix)) else f"{name}{suffix}"
        if ds_entry not in dataset_names:
            dataset_names.append(ds_entry)
        if not name.lower().startswith("lemgendized"):
            lem_name = f"LemGendized{name}"
            if lem_name not in dataset_names:
                dataset_names.append(lem_name)
            lem_entry = f"LemGendized{ds_entry}"
            if lem_entry not in dataset_names:
                dataset_names.append(lem_entry)
    return dataset_names


def _list_images(scan_dir):
    """Image files directly in ``scan_dir``, else every image below it (relative, ``/``-joined)."""
    files = [f for f in os.listdir(scan_dir) if f.lower().endswith(_IMAGE_EXTS)]
    if not files:
        for root_dir, _, sub_files in os.walk(scan_dir):
            for sf in sub_files:
                if sf.lower().endswith(_IMAGE_EXTS):
                    files.append(
                        os.path.relpath(os.path.join(root_dir, sf), scan_dir).replace("\\", "/")
                    )
    return files


def _has_any(path, subdirs):
    """True when any of ``subdirs`` exists under ``path`` (checked in order)."""
    return any(os.path.exists(os.path.join(path, sub)) for sub in subdirs)


def _case_insensitive_entries(root, ds_name):
    """Yield ``root`` entries named ``ds_name`` ignoring case; ``os.listdir`` errors propagate."""
    for item in os.listdir(root):
        if item.lower() == ds_name.lower():
            yield os.path.join(root, item)


def _normalized_name(name):
    """Lower-case ``name`` without separators and the LemGendized/LemGendary prefixes."""
    return (
        name.lower()
        .replace("-", "")
        .replace("_", "")
        .replace("lemgendized", "")
        .replace("lemgendary", "")
    )


def _match_attached(ds_name, attached):
    """Attached manifold whose (parent) folder name contains ``ds_name``, else the first one."""
    target = _normalized_name(ds_name)
    for suffix in ["kaggleready", "large", "mini"]:
        target = target.replace(suffix, "")

    # Exact or fuzzy match against attached manifolds
    for cand_path in attached:
        cand_name = _normalized_name(cand_path.name)
        cand_parent = _normalized_name(cand_path.parent.name)
        if target and (target in cand_name or target in cand_parent):
            return str(cand_path)

    # Dynamic Fallback: Auto-bind attached Kaggle dataset manifold
    logger.info(
        "Auto-bound attached Kaggle dataset '%s' for target '%s'",
        attached[0],
        ds_name,
    )
    return str(attached[0])


class DatasetSourcesMixin:
    """Dataset discovery mixed into ``MultiTaskDataset``."""

    def split_path(self, ds_path, folder_name, fname, ext=""):
        """Locate ``fname`` under ``folder_name``, falling back to the split-specific path."""
        base_name = os.path.splitext(fname)[0] if ext else fname
        suffix = ext if ext else ""
        cands = [
            os.path.join(ds_path, folder_name, self.split, base_name + suffix),
            os.path.join(
                ds_path,
                folder_name,
                ("val" if self.split == "train" else "train"),
                base_name + suffix,
            ),
            os.path.join(ds_path, folder_name, base_name + suffix),
            os.path.join(ds_path, self.split, folder_name, base_name + suffix),
            os.path.join(ds_path, self.split, base_name + suffix),
            os.path.join(ds_path, base_name + suffix),
        ]
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0]

    def load_manifest(self, config):
        """Index every configured dataset's samples into ``all_samples`` and ``samples``."""
        self.all_samples = []
        self.samples = []
        if self.env == "kaggle" and config.get("debug", False):
            _print_kaggle_mounts()

        dataset_names = _dataset_names(self._raw_dataset_names(), _dataset_suffix(self.env, config))
        loaded_paths = set()
        for ds_name in dataset_names:
            ds_path = self.get_dataset_path(ds_name)
            if ds_path is None or ds_path in loaded_paths:
                continue
            loaded_paths.add(ds_path)
            self.path_cache[ds_name] = ds_path
            scan_dir = self._scan_dir(ds_path)
            if not scan_dir or not os.path.exists(scan_dir):
                continue
            for f in _list_images(scan_dir):
                self.all_samples.append((ds_name, f))

        # Kaggle dynamic discovery: Train on whichever datasets are attached in /kaggle/input
        if (self.env == "kaggle" or os.path.exists("/kaggle/input")) and not self.all_samples:
            self._load_attached_manifolds(config, loaded_paths)

        # Container-format fallback: if no directory samples found, try ContainerReader.
        if not self.all_samples:
            self.load_container_samples(loaded_paths)

        self._apply_sample_fraction()

    def _raw_dataset_names(self):
        """Dataset names from the model registry entry, defaulting to the model key."""
        raw_dataset_names = self.model_info.get("datasets", [])
        if isinstance(raw_dataset_names, str):
            raw_dataset_names = [raw_dataset_names]
        elif not isinstance(raw_dataset_names, list):
            raw_dataset_names = []
        if not raw_dataset_names:
            raw_dataset_names = [self.model_key]
        return raw_dataset_names

    def _scan_dir(self, ds_path):
        """Directory whose images index ``ds_path`` for the current split."""
        # 2026: Parameter prediction loads from targets/ (clean source images)
        if self.task_type == "parameter_prediction":
            img_dir = os.path.join(ds_path, "targets", self.split)
            if not os.path.exists(img_dir):
                img_dir = os.path.join(ds_path, "images", self.split)
            return img_dir
        img_dir = os.path.join(ds_path, "images", self.split)
        tgt_dir = os.path.join(ds_path, "targets", self.split)
        return img_dir if os.path.exists(img_dir) else tgt_dir

    def _load_attached_manifolds(self, config, loaded_paths):
        """Index the first image folder of every manifold attached under ``/kaggle/input``."""
        resolver = ManifoldResolver(env=self.env, config=config)
        attached_manifolds = resolver.scan_kaggle_input_manifolds()
        if not attached_manifolds:
            return
        if os.environ.get("RANK", "0") == "0":
            print(
                f"\n[KAGGLE] Auto-discovered {len(attached_manifolds)} attached "
                "dataset manifold(s) in /kaggle/input:"
            )
            for m_path in attached_manifolds:
                print(f"  -> {m_path}")
        for m_path in attached_manifolds:
            m_str = str(m_path)
            loaded_paths.add(m_str)
            ds_key = m_path.name
            self.path_cache[ds_key] = m_str
            for sub in ["images", "targets", "shards", "train", "val"]:
                sub_dir = (
                    os.path.join(m_str, sub, self.split)
                    if os.path.exists(os.path.join(m_str, sub, self.split))
                    else os.path.join(m_str, sub)
                )
                if os.path.exists(sub_dir):
                    items = _list_images(sub_dir)
                    for f in items:
                        self.all_samples.append((ds_key, f))
                    if items:
                        break

    def _apply_sample_fraction(self) -> None:
        """Apply deterministic curriculum sample-fraction slicing to ``self.all_samples``."""
        self.samples = list(self.all_samples)
        if self.is_train and self.sample_fraction < 1.0 and self.samples:
            rng = random.Random(42)
            rng.shuffle(self.samples)
            target_count = max(1, int(len(self.samples) * float(self.sample_fraction)))
            self.samples = self.samples[:target_count]

    def get_dataset_path(self, ds_name):
        """Resolve the on-disk directory for dataset ``ds_name`` in the current environment.

        Exits the process when a Colab run cannot locate a required dataset.
        """
        resolver = ManifoldResolver(env=self.env, config=self.config)
        resolved = resolver.resolve_manifold(ds_name)
        if resolved is not None:
            return str(resolved)

        model_info = self.unified_models.get(self.model_key, {})
        gdrive_id = model_info.get("google_drive_dataset_id")

        if self.env == "colab":
            return self._colab_dataset_path(ds_name, gdrive_id)
        if self.env == "kaggle":
            return self._kaggle_dataset_path(ds_name, resolver)
        return self._local_dataset_path(ds_name, gdrive_id)

    def _colab_dataset_path(self, ds_name, gdrive_id):
        """Colab lookup: downloaded copy, then the Google Drive mount; exits when missing."""
        colab_ds_root = "/content/LemGendaryDatasets"
        found, cand = self._colab_downloaded_path(ds_name, colab_ds_root)
        if found is not None:
            return found
        # Priority 2: Check Google Drive mount
        drive_path = f"/content/drive/MyDrive/LemGendaryDatasets/{ds_name}"
        if os.path.exists(drive_path):
            return drive_path
        if gdrive_id:
            if not os.path.exists(drive_path):
                print(f"\n[ERROR] Colab requires dataset at {cand} or Google Drive at {drive_path}")
                sys.exit(1)
            return drive_path
        # Priority 3: Fallback check anywhere in /content/LemGendaryDatasets
        if os.path.exists(colab_ds_root):
            cand = os.path.join(colab_ds_root, ds_name)
            if os.path.exists(cand):
                return cand
        print(
            f"\n[ERROR] Colab could not locate dataset '{ds_name}' in {colab_ds_root} "
            "or Google Drive."
        )
        sys.exit(1)

    def _colab_downloaded_path(self, ds_name, colab_ds_root):
        """Priority 1 on Colab: ``(path or None, last candidate tried)``.

        The last candidate is what the missing-dataset error message names.
        """
        # Priority 1: Check downloaded Kaggle dataset in /content/LemGendaryDatasets
        cand = os.path.join(colab_ds_root, ds_name)
        if os.path.exists(cand) and (
            os.path.exists(os.path.join(cand, "images"))
            or os.path.exists(os.path.join(cand, "targets"))
            or any(f.endswith((".parquet", ".csv")) for f in os.listdir(cand))
        ):
            return cand, cand
        if hasattr(self, "data_root") and os.path.exists(self.data_root):
            cand = os.path.join(self.data_root, ds_name)
            if os.path.exists(cand):
                return cand, cand
        # Check case-insensitive in colab_ds_root
        if os.path.exists(colab_ds_root):
            try:
                for cand in _case_insensitive_entries(colab_ds_root, ds_name):
                    if os.path.exists(cand):
                        return cand, cand
            except OSError as exc:
                logger.warning(
                    "Case-insensitive scan of %s failed for %s: %s", colab_ds_root, ds_name, exc
                )
        return None, cand

    def _kaggle_dataset_path(self, ds_name, resolver):
        """Kaggle lookup: the symlinked data root, then manifolds attached in /kaggle/input."""
        # Priority 1: Check symlinked data_root (/kaggle/working/LemGendaryDatasets)
        if hasattr(self, "data_root") and os.path.exists(self.data_root):
            found = self._kaggle_data_root_path(ds_name)
            if found is not None:
                return found

        # Priority 2: Scan /kaggle/input for attached manifolds (directory or container format)
        if os.path.exists("/kaggle/input"):
            attached = resolver.scan_kaggle_input_manifolds()
            if attached:
                return _match_attached(ds_name, attached)

        print(f"\n[WARNING] No attached dataset found for '{ds_name}' in Kaggle /kaggle/input!")
        print("Please attach the dataset to your Kaggle Notebook.")
        return None

    def _kaggle_data_root_path(self, ds_name):
        """``ds_name`` (exact, then case-insensitive) under the Kaggle data root, or ``None``."""
        cand = os.path.join(self.data_root, ds_name)
        if _has_any(cand, _KAGGLE_LAYOUT):
            return cand
        try:
            for cand in _case_insensitive_entries(self.data_root, ds_name):
                if _has_any(cand, _KAGGLE_LAYOUT):
                    return cand
        except OSError as exc:
            logger.warning(
                "Case-insensitive scan of %s failed for %s: %s",
                self.data_root,
                ds_name,
                exc,
            )
        return None

    def _local_dataset_path(self, ds_name, gdrive_id):
        """Local lookup: the data root (exact, then case-insensitive), then a gdown download."""
        # Priority 1: Check Local Root
        path = os.path.join(self.data_root, ds_name)
        if _has_any(path, _LOCAL_LAYOUT):
            return path

        if os.path.exists(self.data_root):
            try:
                for cand in _case_insensitive_entries(self.data_root, ds_name):
                    if _has_any(cand, _LOCAL_LAYOUT):
                        return cand
            except OSError as local_err:
                logger.debug("Failed listing local data root: %s", local_err)

        # Priority 2: Use gdown
        if gdrive_id:
            return self._gdown_dataset(ds_name, gdrive_id)
        return None

    def _gdown_dataset(self, ds_name, gdrive_id):
        """Download and extract ``ds_name`` from Google Drive; ``None`` when nothing arrived."""
        print(
            "\n[DATA] Local dataset not found. Falling back to gdown from Google Drive "
            f"(ID: {gdrive_id})..."
        )
        os.makedirs(self.data_root, exist_ok=True)
        dest = os.path.join(self.data_root, f"{ds_name}.zip")
        result = subprocess.run(
            [sys.executable, "-m", "gdown", "--id", gdrive_id, "-O", dest], check=False
        )
        if result.returncode != 0:
            logger.warning("gdown exited with code %d for dataset %s", result.returncode, ds_name)
        if os.path.exists(dest):
            print(f"[DATA] Extracting {dest}...")
            with zipfile.ZipFile(dest, "r") as zip_ref:
                zip_ref.extractall(self.data_root)
            os.remove(dest)
            return os.path.join(self.data_root, ds_name)
        return None
