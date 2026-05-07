import os
import sys
import yaml

# Anchor the search path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from training.notebook_generator import generate_inference_notebook, generate_usage_notebook

def main():
    # 1. Load configuration
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    unified_name = config.get("unified_models", "unified_models_v2.yaml")
    unified_models_path = os.path.join(script_dir, unified_name)
    
    with open(unified_models_path, "r") as f:
        registry = yaml.safe_load(f)

    # 2. Identify target hub
    hub_root = os.path.abspath(os.path.join(script_dir, "..", "LemGendaryModels"))
    print(f"[*] [REFRESH] Target Hub: {hub_root}")

    # 3. Iterate and Regenerate
    for model_key in registry.keys():
        model_dir = os.path.join(hub_root, model_key)
        print(f"[MODEL] Processing {model_key}...")
        
        # Ensure model directory exists in the hub
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Generate Inference/Training Notebook
            generate_inference_notebook(model_key, model_dir, registry, config)
            
            # Generate Usage Notebook
            generate_usage_notebook(model_key, model_dir, registry, config)
            
            print(f"[*] [SUCCESS] {model_key} notebooks are now v10.0 compliant.")
        except Exception as e:
            print(f"[!] [FAILED] {model_key} refresh failed: {e}")

if __name__ == "__main__":
    main()
