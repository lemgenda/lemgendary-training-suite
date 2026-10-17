import os
import shutil

import yaml


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

    class_map = {
        "retinaface": ["face"],
        "retinaface_mobilenet": ["face"],
        "retinaface_resnet": ["face"],
        "yolov8n": ["face", "person", "hand", "eye"],
    }

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ]

    names = class_map.get(model_key, ["object"])
    if model_key == "yolov8n":
        names = list(COCO_CLASSES)

    from training.data.manifold import ManifoldResolver

    resolver = ManifoldResolver(env=config.get("env", "local"), config=config)
    resolved_path = resolver.resolve_manifold(dataset_names[0])

    if resolved_path is not None and resolved_path.exists():
        classes_file = resolved_path / "classes.txt"
        if classes_file.exists():
            try:
                raw_lines = classes_file.read_text(encoding="utf-8").splitlines()
                c_names = [line.strip() for line in raw_lines if line.strip()]
                if c_names and len(c_names) >= 80:
                    names = c_names
                elif c_names and model_key != "yolov8n":
                    names = c_names
            except Exception:
                pass

        if (resolved_path / "images" / "train").exists():
            rel_train = "images/train"
        elif (resolved_path / "train" / "images").exists():
            rel_train = "train/images"
        elif (resolved_path / "train").exists():
            rel_train = "train"
        elif (resolved_path / "images").exists():
            rel_train = "images"
        else:
            rel_train = "."

        if (resolved_path / "images" / "val").exists():
            rel_val = "images/val"
        elif (resolved_path / "val" / "images").exists():
            rel_val = "val/images"
        elif (resolved_path / "val").exists():
            rel_val = "val"
        else:
            rel_val = rel_train

        yolo_cfg = {
            "path": str(resolved_path),
            "train": rel_train,
            "val": rel_val,
            "nc": len(names),
            "names": dict(enumerate(names)),
        }
    else:
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
        if not os.path.exists(os.path.join(abs_data_root, primary_ds)):
            if os.path.exists(os.path.join(abs_data_root, dataset_names[0])):
                primary_ds = dataset_names[0]

        yolo_cfg = {
            "path": abs_data_root,
            "train": os.path.join(primary_ds, "images", "train"),
            "val": os.path.join(primary_ds, "images", "val"),
            "nc": len(names),
            "names": dict(enumerate(names)),
        }

    suite_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(suite_root, ".cache", "yolo_configs")
    os.makedirs(cache_dir, exist_ok=True)
    temp_cfg_path = os.path.join(cache_dir, f"yolo_{model_key}_config.yaml")

    legacy_hub_training_dir = os.path.join(os.path.dirname(suite_root), "LemGendaryModels", model_key, "training")
    if os.path.exists(legacy_hub_training_dir):
        try:
            shutil.rmtree(legacy_hub_training_dir, ignore_errors=True)
        except OSError:
            pass

    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    print(f"[YOLO GEN] Dynamic config materialized for {model_key} with {len(names)} classes at {temp_cfg_path}.")
    return temp_cfg_path

