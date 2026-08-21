import os
import yaml

def generate_yolo_yaml(config, model_key, unified_models_registry):
    """
    2026 Dynamic YOLO Config Generator (v2.0)
    Resolves physical dataset paths and class counts from the unified registries.
    """
    model_info = unified_models_registry.get(model_key, {})
    dataset_names = model_info.get("datasets", [])
    
    if not dataset_names:
        return None
        
    data_root = config.get("paths", {}).get("datasets_root", "data/datasets")
    # Resolve relative to project root
    abs_data_root = os.path.abspath(data_root)
    
    # 2026: Universal Class Mapping for LemGendary Detection Array
    # This ensures consistency across RetinaFace and YOLOv8 backbones
    class_map = {
        "retinaface_mobilenet": ["face"],
        "retinaface_resnet": ["face"],
        "yolov8n": ["face", "person", "hand", "eye"] # Expanded Master Detection Set
    }
    
    names = class_map.get(model_key, ["object"])
    if model_key == "yolov8n":
        while len(names) < 80:
            names.append(f"class_{len(names)}")
    
    # Use the FIRST dataset listed as the primary path anchor
    suffix = "KaggleReady" if config.get("env") == 'kaggle' else config.get("execution", {}).get("suffixes", {}).get(config.get("execution", {}).get("mode", "training"), "")
    primary_ds = dataset_names[0] if (suffix and dataset_names[0].endswith(suffix)) else f"{dataset_names[0]}{suffix}"
    train_path = os.path.join(abs_data_root, primary_ds, "images", "train")
    val_path = os.path.join(abs_data_root, primary_ds, "images", "val")
    
    yolo_cfg = {
        "path": abs_data_root,
        "train": os.path.join(primary_ds, "images", "train"),
        "val": os.path.join(primary_ds, "images", "val"),
        "nc": len(names),
        "names": {i: name for i, name in enumerate(names)}
    }
    
    temp_cfg_path = os.path.join("data", f"yolo_{model_key}_config.yaml")
    os.makedirs("data", exist_ok=True)
    
    with open(temp_cfg_path, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)
        
    print(f"[YOLO GEN] Dynamic config materialized for {model_key} with {len(names)} classes.")
    return temp_cfg_path
