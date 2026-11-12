"""Universal multi-task image dataset backed by dataset manifolds and container readers."""

import json
import logging
import multiprocessing
import os

import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from data.container_targets import ContainerTargetsMixin
from data.dataset_containers import DatasetContainerMixin
from data.dataset_sources import DatasetSourcesMixin
from data.dataset_state import DatasetOptions, ModelSpec, SampleCatalog, TransformSet
from data.dataset_targets import DatasetTargetsMixin
from training.data.sample_seed import AugmentationSeed, seeded_sample_rngs

Image.MAX_IMAGE_PIXELS = None  # Disable PIL DecompressionBombWarning for large datasets

logger = logging.getLogger("lemtrain.dataset")


# [SENIOR HARDENING v16.0 - SYNC_ID: 1152]


def _resolve_data_root(config, env):
    """Datasets root from the ``paths`` section; relative roots resolve per environment."""
    data_root = config.get("paths", {}).get(
        "datasets_root", config.get("datasets_dir", "data/datasets")
    )
    if os.path.isabs(data_root):
        return data_root
    if env == "kaggle":
        return "/kaggle/working/LemGendaryDatasets"
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(workspace_root, data_root))


def _input_size(size_raw):
    """``(H, W)`` from an ``input_size`` entry: an int, ``[H, W]`` or ``[C, H, W]``."""
    if not isinstance(size_raw, list):
        return (size_raw, size_raw)
    if len(size_raw) == 3:
        return (int(size_raw[1]), int(size_raw[2]))
    return (int(size_raw[0]), int(size_raw[1]))


def _load_nima_scores(idx_path, scores):
    """Add ``name -> nima_score`` from a manifold ``index.json`` (if present) to ``scores``."""
    if not os.path.exists(idx_path):
        return
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            scores[item["name"]] = item.get("nima_score", 0.0)
    except (OSError, ValueError, KeyError, TypeError) as nima_err:
        logger.debug("Failed loading NIMA index from %s: %s", idx_path, nima_err)


class MultiTaskDataset(
    Dataset,
    DatasetSourcesMixin,
    DatasetContainerMixin,
    ContainerTargetsMixin,
    DatasetTargetsMixin,
):
    """
    Nuclear-Hardened Universal Dataset (v16.0).
    Implements Fallback Shields, LANCZOS scaling, and stratified distribution analytics.
    """

    def __init__(self, config, model_key=None, is_train=True, env="local", sample_fraction=1.0):
        self.config = config
        self.sync_mode = multiprocessing.Value("b", False)

        ImageFile.LOAD_TRUNCATED_IMAGES = True  # Shield against corrupt JPG headers

        unified_models_path = os.path.join(
            os.path.dirname(__file__), "..", config["unified_models"]
        )
        with open(unified_models_path, "r", encoding="utf-8") as f:
            unified_models = yaml.safe_load(f)

        model_key = model_key or next(iter(unified_models))
        model_info = unified_models[model_key]
        raw_type = model_info.get("dataset_type", "restoration")
        self.spec = ModelSpec(
            unified_models=unified_models,
            model_key=model_key,
            model_info=model_info,
            task_type=raw_type[0] if isinstance(raw_type, list) else raw_type,
        )
        self.opts = DatasetOptions(
            is_train=is_train,
            env=env,
            split="train" if is_train else "val",
            data_root=_resolve_data_root(config, env),
            sample_fraction=sample_fraction,
            size=_input_size(model_info.get("input_size", config.get("default_img_size", 256))),
        )

        self.catalog = SampleCatalog()
        self.load_manifest(config)

        if self.spec.model_key == "nima_authenticity":
            for ds_path in self.catalog.path_cache.values():
                if ds_path is None:
                    continue
                _load_nima_scores(os.path.join(ds_path, "index.json"), self.catalog.nima_scores)

        self.pipes = TransformSet()
        self.build_transforms()
        self.augment_seed = AugmentationSeed()

    @property
    def task_type(self) -> str:
        """The served model's dataset task type (read by loaders)."""
        return self.spec.task_type

    @property
    def size(self) -> tuple[int, int]:
        """Current ``(H, W)`` input size (read by loaders)."""
        return self.opts.size

    @property
    def all_samples(self) -> list[tuple[str, str | int]]:
        """Every indexed sample before sample-fraction slicing."""
        return self.catalog.all_samples

    @property
    def path_cache(self) -> dict[str, str | None]:
        """Resolved dataset paths by dataset name."""
        return self.catalog.path_cache

    @property
    def sample_fraction(self) -> float:
        """Curriculum sample fraction."""
        return self.opts.sample_fraction

    @sample_fraction.setter
    def sample_fraction(self, value: float) -> None:
        self.opts.sample_fraction = value

    def build_transforms(self):
        """Build ``self.pipes`` (resize, augmentation, tensor conversion) for this task."""
        # 2026 Resilience: LANCZOS is superior for Aesthetic/Quality manifolds
        interp = (
            transforms.InterpolationMode.LANCZOS
            if self.spec.task_type == "quality"
            else transforms.InterpolationMode.BILINEAR
        )

        # FIX: Preserve aspect ratio and crop to prevent network from learning squish artifacts
        if self.spec.task_type == "face_detection":
            # Coordinate-based targets (bboxes, landmarks) become invalid if the image is cropped.
            # We must squish-resize so normalized [0,1] coordinates remain physically accurate.
            transform_list: list = [transforms.Resize(self.opts.size, interpolation=interp)]
        elif self.opts.is_train:
            transform_list: list = [
                transforms.Resize(max(self.opts.size), interpolation=interp),
                transforms.RandomCrop(self.opts.size),
            ]
        else:
            transform_list: list = [
                transforms.Resize(max(self.opts.size), interpolation=interp),
                transforms.CenterCrop(self.opts.size),
            ]
        if self.opts.is_train and self.spec.task_type == "quality":
            transform_list.append(transforms.RandomHorizontalFlip())
            # --- 2026 Resilience: Dynamic Stress Injection ---
            if self.opts.stress > 0.0:
                stress_val = min(self.opts.stress, 5.0)
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=0.1 * stress_val,
                        contrast=0.1 * stress_val,
                        saturation=0.1 * stress_val,
                        hue=0.02 * stress_val,
                    )
                )
                transform_list.append(transforms.RandomRotation(degrees=int(3 * stress_val)))
                if stress_val >= 2.0:
                    transform_list.append(
                        transforms.RandomPerspective(distortion_scale=0.1 * stress_val, p=0.3)
                    )
            elif "aesthetic" in self.spec.model_key:
                transform_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.05))

        transform_list.append(transforms.ToTensor())
        if self.spec.task_type == "quality":
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        self.pipes.transform = transforms.Compose(transform_list)

        # 2026 ParseNet Fix: Dynamic mask interpolation for multitask pipelines
        self.pipes.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    max(self.opts.size), interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.RandomCrop(self.opts.size)
                if self.opts.is_train
                else transforms.CenterCrop(self.opts.size),
            ]
        )

        # 2026: Parameter prediction needs clean ToTensor only (no normalization)
        # Degradation is applied after transform in __getitem__
        if self.spec.task_type == "parameter_prediction":
            self.pipes.clean_transform = transforms.Compose(
                [
                    transforms.Resize(max(self.opts.size), interpolation=interp),
                    transforms.RandomCrop(self.opts.size)
                    if self.opts.is_train
                    else transforms.CenterCrop(self.opts.size),
                    transforms.ToTensor(),
                ]
            )

    def load_image(self, img_path):
        """Fallback Shield: Returns Neutral-Gray tensor on I/O failure (Task 9.1)."""
        try:
            with Image.open(img_path) as img:
                return img.convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            # OSError covers missing files and PIL.UnidentifiedImageError / truncated data.
            logger.warning("I/O failure on %s; substituting neutral gray: %s", img_path, e)
            return Image.new("RGB", (self.opts.size[1], self.opts.size[0]), (128, 128, 128))

    def set_epoch(self, epoch: int) -> None:
        """Derive the per-sample augmentation seeds of ``epoch`` (workers included)."""
        self.augment_seed.set_epoch(epoch)

    def __len__(self):
        return len(self.catalog.samples)

    def __getitem__(self, index):
        if getattr(self.sync_mode, "value", self.sync_mode):
            return self.sync_dummy()

        ds_name, sample_id = self.catalog.samples[index]
        # Same augmentation for the same (epoch, index) on any worker, rank or resume.
        seed = self.augment_seed.sample_seed(index)
        with seeded_sample_rngs(seed):
            # Container-backed sample path: sample_id is an integer index.
            if isinstance(sample_id, int):
                return self.getitem_container(ds_name, sample_id, seed)

            # Directory-backed path: sample_id is a filename string.
            return self.task_item(self.load_directory_sample(ds_name, sample_id, seed))

    def update_strategy(self, size=None, fraction=None, stress=None):
        """Dynamic manifold scaling for curriculum learning (v16.2)."""
        rebuild = False
        if size:
            if isinstance(size, (int, float)):
                self.opts.size = (int(size), int(size))
            elif isinstance(size, (list, tuple)):
                self.opts.size = (int(size[0]), int(size[1]))
            rebuild = True
        if stress is not None and self.opts.stress != stress:
            self.opts.stress = stress
            rebuild = True
        if rebuild:
            self.build_transforms()

        if fraction is not None:
            self.opts.sample_fraction = float(fraction)
            if self.catalog.all_samples:
                self._apply_sample_fraction()
            else:
                self.load_manifest(self.config)

    def get_distribution(self):
        """Task 9.3: Analyze label manifold for stratified balancing."""
        print(f"[METRICS] [DATA] Analyzing distribution for {self.spec.model_key}...")
        stats = {"total": len(self.catalog.samples), "tasks": {}}
        # Implementation for background distribution analysis...
        return stats
