"""Generate Ultralytics YOLO dataset YAML configs from the unified model registry."""

import logging
import os
import shutil

import yaml

from training.data.manifold import ManifoldResolver

logger = logging.getLogger("lemtrain.data.yolo_config_gen")

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


_CLASS_MAP: dict[str, list[str]] = {
    "retinaface": ["face"],
    "retinaface_mobilenet": ["face"],
    "retinaface_resnet": ["face"],
    "yolov8n": ["face", "person", "hand", "eye"],
}
"""Default class names per model key (``yolov8n`` uses the COCO classes)."""

_TRAIN_DIRS = ("images/train", "train/images", "train", "images")
_VAL_DIRS = ("images/val", "val/images", "val")


def _default_names(model_key):
    names = _CLASS_MAP.get(model_key, ["object"])
    if model_key == "yolov8n":
        names = list(COCO_CLASSES)
    return names


def _manifold_names(resolved_path, model_key, names):
    """Class names from the manifold's ``classes.txt`` when it is usable, else ``names``."""
    classes_file = resolved_path / "classes.txt"
    if classes_file.exists():
        try:
            raw_lines = classes_file.read_text(encoding="utf-8").splitlines()
            c_names = [line.strip() for line in raw_lines if line.strip()]
            if c_names and len(c_names) >= 80 or c_names and model_key != "yolov8n":
                names = c_names
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s; keeping default class names: %s", classes_file, exc)
    return names


def _first_existing(root, candidates, default):
    """First of ``candidates`` (relative paths) that exists below ``root``, else ``default``."""
    for rel in candidates:
        if (root / rel).exists():
            return rel
    return default


def _manifold_cfg(resolved_path, names):
    """Ultralytics data config for a resolved manifold folder."""
    rel_train = _first_existing(resolved_path, _TRAIN_DIRS, ".")
    rel_val = _first_existing(resolved_path, _VAL_DIRS, rel_train)
    return {
        "path": str(resolved_path),
        "train": rel_train,
        "val": rel_val,
        "nc": len(names),
        "names": dict(enumerate(names)),
    }


def _fallback_cfg(config, dataset_names, abs_data_root, names):
    """Data config below ``datasets_root`` when the manifold could not be resolved."""
    exec_mode = config.get("execution", {}).get("mode", "training")
    suffix = (
        "KaggleReady"
        if config.get("env") == "kaggle"
        else config.get("execution", {}).get("suffixes", {}).get(exec_mode, "")
    )
    primary_ds = (
        dataset_names[0]
        if (suffix and dataset_names[0].endswith(suffix))
        else f"{dataset_names[0]}{suffix}"
    )
    if not os.path.exists(os.path.join(abs_data_root, primary_ds)) and os.path.exists(
        os.path.join(abs_data_root, dataset_names[0])
    ):
        primary_ds = dataset_names[0]

    return {
        "path": abs_data_root,
        "train": os.path.join(primary_ds, "images", "train"),
        "val": os.path.join(primary_ds, "images", "val"),
        "nc": len(names),
        "names": dict(enumerate(names)),
    }


def _remove_legacy_training_dir(suite_root, model_key):
    legacy_hub_training_dir = os.path.join(
        os.path.dirname(suite_root), "LemGendaryModels", model_key, "training"
    )
    if os.path.exists(legacy_hub_training_dir):
        try:
            shutil.rmtree(legacy_hub_training_dir)
        except OSError as exc:
            logger.warning(
                "Could not remove legacy hub training dir %s: %s", legacy_hub_training_dir, exc
            )


def generate_yolo_yaml(config, model_key, unified_models_registry):
    """2026 Dynamic YOLO Config Generator (v2.0).

    Resolves physical dataset paths and class counts from the unified registries.
    """
    model_info = unified_models_registry.get(model_key, {})
    dataset_names = model_info.get("datasets", [])

    if not dataset_names:
        return None

    data_root = config.get("paths", {}).get("datasets_root", "data/datasets")
    abs_data_root = os.path.abspath(data_root)
    names = _default_names(model_key)

    resolver = ManifoldResolver(env=config.get("env", "local"), config=config)
    resolved_path = resolver.resolve_manifold(dataset_names[0])

    if resolved_path is not None and resolved_path.exists():
        names = _manifold_names(resolved_path, model_key, names)
        yolo_cfg = _manifold_cfg(resolved_path, names)
    else:
        yolo_cfg = _fallback_cfg(config, dataset_names, abs_data_root, names)

    suite_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(suite_root, ".cache", "yolo_configs")
    os.makedirs(cache_dir, exist_ok=True)
    temp_cfg_path = os.path.join(cache_dir, f"yolo_{model_key}_config.yaml")
    _remove_legacy_training_dir(suite_root, model_key)

    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    print(
        f"[YOLO GEN] Dynamic config materialized for {model_key} with {len(names)} classes "
        f"at {temp_cfg_path}."
    )
    return temp_cfg_path
