import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn

# --- 2026 Hardware Acceleration & Stability Patch ---
# Anchor the search path to the parent directory to allow root module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Increase recursion limit for exceptionally deep architectures (NIMA/Restorers)
sys.setrecursionlimit(2000)

class SoftmaxWrapper(nn.Module):
    def __init__(self, inner_model, temperature=1.0):
        super().__init__()
        self.inner_model = inner_model
        self.temperature = temperature
    def forward(self, x):
        logits = self.inner_model(x)
        return torch.nn.functional.softmax(logits / self.temperature, dim=1)

def main():
    parser = argparse.ArgumentParser(description="LemGendary SOTA Exporter: Checkpoint to Standalone PyTorch")
    parser.add_argument("--model", type=str, required=True, help="Model key from unified_models.yaml")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive prompts for automated 2026 pipelines")
    args = parser.parse_args()

    print(f"\nInitializing On-Demand SOTA PT Exporter for model: {args.model}")
    
    # 1. Environment Discovery (Hierarchical Path Resolution)
    config_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(config_path):
        print(f" Error: config.yaml not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    unified_models_name = config.get("unified_models", "unified_models.yaml")
    unified_models_path = os.path.join(project_root, unified_models_name)
    if not os.path.exists(unified_models_path):
        print(f" Error: {unified_models_path} not found.")
        return
        
    with open(unified_models_path, 'r') as f:
        unified_models_registry = yaml.safe_load(f)

    model_info = unified_models_registry.get(args.model)
    if not model_info:
        print(f"Error: Model '{args.model}' not found in registry.")
        return

    # 2. Architecture Instantiation
    from models.factory import get_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [ARCH] Instantiating architecture for {args.model} on {device}...")
    try:
        model = get_model(args.model, config).to(device)
    except Exception as e:
        print(f" Error during instantiation: {e}")
        return

    # 3. Checkpoint Discovery
    ckpt_dir_rel = config.get("checkpoint_dir", "checkpoints")
    ckpt_dir = os.path.normpath(os.path.join(project_root, ckpt_dir_rel))
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    
    if not os.path.exists(ckpt_path):
        print(f" Error: SOTA Checkpoint not found at {ckpt_path}")
        return

    print(f" [LOAD] Extracting weights from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            # Handle missing softmax_temp by using strict=False
            model.load_state_dict(ckpt['model_state'], strict=False)
            print(f"   -> Epoch: {ckpt.get('epoch', 'N/A')} | Quality: {ckpt.get('sota_metric', 'N/A')}")
        else:
            model.load_state_dict(ckpt)
            print("   -> Successfully loaded raw weights.")
    except Exception as e:
        print(f" Error: Failed to load state dictionary: {e}")
        return
    
    model.eval()

    # 3.5 Production Wrapping (Probabilities for Quality Manifolds)
    if model_info.get("dataset_type") == "quality":
        temp = getattr(model, "softmax_temp", torch.tensor(1.0)).item()
        if temp == 1.0:
            stab = model_info.get("stabilizers", {})
            temp = stab.get("softmax_temp", 1.0)
        
        print(f" [WRAP] Applying production Softmax wrapper with Temp={temp}")
        model = SoftmaxWrapper(model, temperature=temp)
        model.eval()

    # 4. Production Synchronization
    model_filename = model_info.get("filename", args.model)
    base_name = f"LemGendary{model_filename}"
    production_dir_rel = os.path.join(config.get("export_dir", "../LemGendaryModels"), args.model)
    production_dir = os.path.normpath(os.path.join(project_root, production_dir_rel))
    os.makedirs(production_dir, exist_ok=True)
    
    target_path = os.path.join(production_dir, f"{base_name}.pt")
    
    if os.path.exists(target_path):
        if args.yes:
            print(f"   -> [OVERWRITE] Non-interactive bypass active for {base_name}.pt.")
        else:
            print(f"\n  [WARNING] Production artifact '{target_path}' already exists.")
            ans = input("  Do you want to OVERWRITE this standalone model? [y/N]: ")
            if ans.lower().strip() != 'y':
                print("  Export aborted by user.")
                return

    print(f" [EXPORT] Saving standalone PyTorch model object to {target_path}...")
    try:
        # Saving the full model (Architecture + Weights)
        torch.save(model, target_path)
        print(" [SUCCESS] Standalone SOTA model is now production-ready.")
        print(f"   -> Usage: model = torch.load('{target_path}')")
    except Exception as e:
        print(f" Error during export: {e}")

if __name__ == "__main__":
    main()
