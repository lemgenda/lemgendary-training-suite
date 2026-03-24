import os
import yaml

def generate_yolo_yaml(config_data, model_key, unified_models, unified_data):
    """
    Generates a YOLO-compatible dataset.yaml dynamically for a specific LemGendized dataset
    assigned to a YOLO model.
    """
    model_info = unified_models.get(model_key)
    if not model_info:
        raise ValueError(f"Model {model_key} not found in unified_models")
    
    datasets = model_info.get("datasets", [])
    if not datasets:
        raise ValueError(f"No datasets assigned to {model_key}")
        
    # We take the first dataset assigned to the model for YOLO training, assuming one primary dataset
    ds_name = datasets[0]
    
    # Find dataset path in unified_data
    # We must scan categories in unified_data (restoration, detection, etc.)
    ds_path = ""
    for cat, dsets in unified_data.get("datasets", {}).items():
        if ds_name in dsets:
            ds_path = str(dsets[ds_name].get("path", ""))
            break
            
    if not ds_path:
        # Fallback heuristic
        ds_path = str(os.path.abspath(os.path.join(config_data.get("datasets_dir", "../data/datasets"), ds_name)))
        
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset path {ds_path} does not exist.")

    # Read classes.txt if present to get class names
    classes_file = os.path.join(ds_path, "classes.txt")
    names = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    names[i] = line
    else:
        # Fallback to generic class if missing
        names = {0: "object"}
        
    yaml_content = {
        "path": ds_path,
        "train": "images/train",
        "val": "images/val" if os.path.exists(os.path.join(ds_path, "images", "val")) else "images/train",
        "names": names
    }
    
    out_yaml = os.path.join(ds_path, "yolo_dataset.yaml")
    with open(out_yaml, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    return out_yaml
